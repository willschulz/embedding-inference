"""
GPU-accelerated embedding API with dynamic batching.

Change MODEL_NAME (or set the env var) to swap the model.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Configuration (single place to change the model) ──────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "BAAI/bge-m3")
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "64"))
MAX_WAIT_MS: float = float(os.getenv("MAX_WAIT_MS", "10"))
PORT: int = int(os.getenv("PORT", "8001"))

log = logging.getLogger("embedder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── Dynamic Batcher ──────────────────────────────────────────────────────
class _Req:
    """Lightweight container for a pending embed request."""
    __slots__ = ("texts", "normalize", "future")

    def __init__(self, texts: List[str], normalize: bool, future: asyncio.Future):
        self.texts = texts
        self.normalize = normalize
        self.future = future


class DynamicBatcher:
    """
    Collects incoming requests for up to *max_wait_ms* or until
    *max_batch* texts have accumulated, then encodes the entire
    mega-batch in one model forward pass.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        max_batch: int = 64,
        max_wait_ms: float = 10,
    ):
        self.model = model
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[_Req] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._loop())
        log.info(
            "Dynamic batcher started  max_batch=%d  max_wait=%.1f ms",
            self.max_batch,
            self.max_wait_ms,
        )

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, texts: List[str], normalize: bool) -> torch.Tensor:
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(_Req(texts, normalize, fut))
        return await fut

    # ── internal ──────────────────────────────────────────────────────
    async def _loop(self):
        while True:
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Batcher step error; retrying in 100 ms")
                await asyncio.sleep(0.1)

    async def _step(self):
        # Block until the first request arrives
        first = await self._queue.get()
        batch: List[_Req] = [first]
        n_texts = len(first.texts)

        # Gather more requests within the time window
        deadline = time.monotonic() + self.max_wait_ms / 1000.0
        while n_texts < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(req)
                n_texts += len(req.texts)
            except asyncio.TimeoutError:
                break

        # Flatten texts and record slices
        all_texts: List[str] = []
        slices: list = []
        offset = 0
        for r in batch:
            slices.append((offset, offset + len(r.texts), r.normalize, r.future))
            all_texts.extend(r.texts)
            offset += len(r.texts)

        log.info("Encoding batch: %d request(s), %d text(s)", len(batch), len(all_texts))

        # Run encoding in the default thread-pool executor
        # (PyTorch releases the GIL during CUDA kernels)
        loop = asyncio.get_running_loop()
        try:
            embeddings = await loop.run_in_executor(None, self._encode, all_texts)
        except Exception as exc:
            for *_, fut in slices:
                if not fut.done():
                    fut.set_exception(exc)
            return

        # Fan results back to individual request futures
        for start, end, normalize, fut in slices:
            try:
                e = embeddings[start:end]
                if normalize:
                    e = torch.nn.functional.normalize(e, p=2, dim=1)
                if not fut.done():
                    fut.set_result(e)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)

    @torch.inference_mode()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=self.max_batch,
            )


# ── FastAPI application ──────────────────────────────────────────────────
batcher: Optional[DynamicBatcher] = None
_model_info: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher, _model_info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model %s on %s …", MODEL_NAME, device)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    # Warm-up pass (triggers CUDA lazy init + JIT)
    model.encode(["warmup"], convert_to_tensor=True, show_progress_bar=False)
    dim = model.get_sentence_embedding_dimension()
    log.info("Model ready in %.1f s — dim=%d", time.time() - t0, dim)
    _model_info = {"model": MODEL_NAME, "device": device, "dim": dim}
    batcher = DynamicBatcher(model, MAX_BATCH_SIZE, MAX_WAIT_MS)
    await batcher.start()
    yield
    await batcher.stop()


app = FastAPI(title="Embedder API", lifespan=lifespan)


# ── Schemas ──────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"ready": batcher is not None, **_model_info}


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if batcher is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not req.texts:
        return EmbedResponse(embeddings=[], dim=_model_info["dim"])
    embs = await batcher.submit(req.texts, req.normalize)
    return EmbedResponse(embeddings=embs.tolist(), dim=embs.shape[1])


# ── Entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, workers=1, log_level="info")
