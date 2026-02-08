# Embedder — GPU-Accelerated Embedding API

A high-throughput text embedding service running **BAAI/bge-m3** (1024-dim) on an NVIDIA L4 GPU, served via FastAPI with server-side dynamic batching.

## Network Address

```
http://embedder-build-v1.manx-celsius.ts.net:8001
```

Accessible from any node on the Tailnet. Substitute the hostname if the VM is renamed.

---

## API Reference

### `GET /health`

Returns service readiness and model metadata.

```bash
curl -sS http://embedder-build-v1.manx-celsius.ts.net:8001/health | jq .
```

```json
{
  "ready": true,
  "model": "BAAI/bge-m3",
  "device": "cuda",
  "dim": 1024
}
```

### `POST /embed`

Encode one or more texts into dense embedding vectors.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `texts` | `string[]` | *(required)* | List of texts to embed |
| `normalize` | `bool` | `true` | L2-normalize the output vectors (required for cosine similarity) |

**Request:**

```bash
curl -sS -X POST http://embedder-build-v1.manx-celsius.ts.net:8001/embed \
  -H 'Content-Type: application/json' \
  -d '{
    "texts": ["How do I reset my password?", "Password recovery instructions"],
    "normalize": true
  }' | jq '{dim, n: (.embeddings | length), preview: .embeddings[0][:5]}'
```

**Response:**

```json
{
  "embeddings": [[0.012, -0.034, ...], [0.008, 0.021, ...]],
  "dim": 1024
}
```

---

## Client Best Practices

### 1. Batch your texts in each request

The single most important optimization. The server encodes all texts in a request as one GPU operation — there is **no per-item HTTP overhead** within a batch. Send as many texts per request as you reasonably can.

```python
import requests

# ✅ Good — one request with 100 texts
texts = ["text 1", "text 2", ..., "text 100"]
resp = requests.post(BASE + "/embed", json={"texts": texts})
embeddings = resp.json()["embeddings"]  # len == 100

# ❌ Bad — 100 separate requests with 1 text each
for t in texts:
    resp = requests.post(BASE + "/embed", json={"texts": [t]})
```

**Guideline:** aim for batches of **16–64 texts** per request. Larger is fine (the server will chunk internally), but you'll see diminishing returns past ~256 and increasing request latency.

### 2. Fire concurrent requests for maximum throughput

The server implements **dynamic batching**: requests arriving within a ~10 ms window are merged into a single GPU forward pass. To exploit this, send requests concurrently rather than sequentially.

```python
import asyncio
import aiohttp

BASE = "http://embedder-build-v1.manx-celsius.ts.net:8001"

async def embed_batch(session, texts):
    async with session.post(f"{BASE}/embed", json={"texts": texts}) as r:
        return (await r.json())["embeddings"]

async def embed_all(text_batches):
    async with aiohttp.ClientSession() as session:
        tasks = [embed_batch(session, batch) for batch in text_batches]
        results = await asyncio.gather(*tasks)
    return [vec for batch in results for vec in batch]

# Split 1000 texts into chunks of 64 and fire them all concurrently
chunks = [all_texts[i:i+64] for i in range(0, len(all_texts), 64)]
embeddings = asyncio.run(embed_all(chunks))
```

### 3. Keep `normalize: true` for similarity search

BGE-M3 embeddings should be L2-normalized for cosine similarity, dot-product ranking, and nearest-neighbor search. This is the default. Only set `normalize: false` if you need raw unnormalized vectors for a custom scoring function.

### 4. Full Python client example

```python
"""Minimal embedding client."""
import requests

EMBEDDER = "http://embedder-build-v1.manx-celsius.ts.net:8001"

def health():
    return requests.get(f"{EMBEDDER}/health").json()

def embed(texts: list[str], normalize: bool = True) -> list[list[float]]:
    resp = requests.post(
        f"{EMBEDDER}/embed",
        json={"texts": texts, "normalize": normalize},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]

# Usage
vecs = embed(["What is the meaning of life?", "42"])
print(f"{len(vecs)} vectors, dim={len(vecs[0])}")
```

### 5. curl one-liners

```bash
# Health check
curl -sS http://embedder-build-v1.manx-celsius.ts.net:8001/health | jq .

# Embed two texts
curl -sS -X POST http://embedder-build-v1.manx-celsius.ts.net:8001/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello world","embedding test"], "normalize":true}' | jq .

# Just get the dimensionality and count
curl -sS -X POST http://embedder-build-v1.manx-celsius.ts.net:8001/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["test"]}' | jq '{dim, count: (.embeddings | length)}'
```

---

## Server Administration

### Service management

```bash
sudo systemctl status embedder      # check status
sudo systemctl restart embedder     # restart
sudo journalctl -u embedder -f      # live logs
docker logs -f embedder             # container logs (equivalent)
```

### Rebuild after code changes

```bash
cd ~/embedder
docker compose build
sudo systemctl restart embedder
```

### Change the model

Edit the `MODEL_NAME` env var in `docker-compose.yml`, rebuild the image (to bake in the new weights), and restart:

```bash
# 1. Edit docker-compose.yml → MODEL_NAME=your/new-model
# 2. Rebuild (re-downloads weights into the image)
docker compose build --no-cache
# 3. Restart
sudo systemctl restart embedder
```

### Tuning

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `BAAI/bge-m3` | HuggingFace model ID |
| `MAX_BATCH_SIZE` | `64` | Max texts merged into one GPU forward pass |
| `MAX_WAIT_MS` | `10` | Time window (ms) to collect concurrent requests before dispatching |
| `PORT` | `8001` | HTTP listen port |

These are set in `docker-compose.yml` under `environment`.

---

## Architecture

```
Client requests (HTTP)
        │
        ▼
   FastAPI (uvicorn, 1 worker)
        │
        ▼
   DynamicBatcher
   ┌──────────────────────────────────┐
   │ async queue collects requests    │
   │ waits up to MAX_WAIT_MS or      │
   │ MAX_BATCH_SIZE texts, then      │
   │ encodes everything in one call  │
   └──────────────────────────────────┘
        │
        ▼
   SentenceTransformer.encode()
   (FP16 autocast, inference_mode, CUDA)
        │
        ▼
   NVIDIA L4 GPU (24 GB VRAM, ~3 GB used by bge-m3)
```

Model weights are baked into the Docker image at build time for instant cold starts — no network download required at boot.
