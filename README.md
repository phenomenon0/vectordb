# vectordb — Local/Hcloud Design (CPU Fast Path)

Goal: run embeddings locally (Hetzner/local CPU) with ONNXRuntime-Go, feed the flat-buffer store, and serve RAG without external APIs.

## Choices
- Model: `bge-small-en` (384d) ONNX FP16 (fast, ~3–8 ms/query on modern CPU).
- Runtime: `github.com/microsoft/onnxruntime-go`. Set session options to use all cores, disable arena growth pauses.
- Hardware: Hetzner CX32/CX42 (8–16 vCPU) or similar local box; no GPU required.
- Store: existing contiguous `VectorStore` (keeps memory hot and scans fast).

## Setup
```bash
# Download model
wget https://huggingface.co/BAAI/bge-small-en/resolve/main/model.onnx -O models/bge-small-en.onnx

# Go deps
go get github.com/microsoft/onnxruntime-go
```

## Embedding Wiring (replace hash embedder in `main.go`)
```go
// Initialize once
sess, _ := onnx.NewSession("models/bge-small-en.onnx", nil)

type Embedder struct {
    sess *onnx.Session
}

func (e *Embedder) Embed(text string) ([]float32, error) {
    // Tokenize text -> ids/attn (use a small tokenizer library or precompute via fast BPE)
    // Build input tensors and run
    outputs, err := e.sess.Run(map[string]*onnx.Value{
        "input_ids":     idsTensor,
        "attention_mask": maskTensor,
    })
    if err != nil { return nil, err }
    vec := outputs[0].Float32s()
    // L2 normalize
    var norm float32
    for _, v := range vec { norm += v * v }
    norm = float32(math.Sqrt(float64(norm)))
    for i := range vec { vec[i] /= norm }
    return vec, nil
}
```

Use the embedder in:
- Ingestion: embed document chunks, `store.Add(vec)`, keep a parallel slice for metadata/IDs (current code uses a fast hash embedder by default—swap in your ONNX embedder).
- Query: embed user query, then `store.Search(vec, k)`.

## Reranker
- Recommended: `bge-reranker-base` (cross-encoder) ONNX for CPU; swap into `Reranker` implementation in `main.go`.
- Interim: the code uses `SimpleReranker` (cosine over embeddings). Replace with an ONNX cross-encoder reranker when ready.

## Concurrency/Batching
- Batch small (8–16) per inference call to amortize runtime overhead.
- Use a worker pool that accepts texts, batches them, runs one session call, returns vectors.
- Set ONNX session options: `IntraOpNumThreads = runtime.NumCPU()`, `InterOpNumThreads = 1`.

## HTTP API (minimal)
Server starts on `:8080` by default.

- `POST /insert`
  ```json
  {"doc": "text to index", "id": "optional-custom-id", "meta": {"tag": "code"}, "upsert": true}
  ```
  Returns `{ "id": "<assigned>" }`. Embeds, indexes, and persists to `vectordb/index.gob`.

- `POST /query`
  ```json
  {
    "query": "what is locality?",
    "top_k": 3,
    "mode": "ann",
    "meta": {"tag": "code"},           // AND filter
    "meta_any": [{"team": "a"}],       // OR filter
    "meta_not": {"env": "dev"},        // NOT filter
    "collection": "default",
    "offset": 0,
    "limit": 3,
    "include_meta": true
  }
  ```
  Embeds query, searches (ANN by default, set `"mode":"scan"` to brute-force), reranks, returns:
  ```json
  {"ids":["..."], "docs": ["..."], "scores": [0.9], "meta":[{"tag":"code"}], "stats": "Rerank completed in ..."}
  ```

- `POST /delete`
  ```json
  {"id": "doc-123"}
  ```
  Marks a document deleted (tombstone; HNSW is not compacted).

- `GET /health`
  Returns basic stats (`total`, `active`, `deleted`, `hnsw_ids`).

## Running
```bash
go run ./vectordb
# HTTP API on :8080; RAG demo also starts in main()
```

## Swap in real models
- Embedder: replace `HashEmbedder` with ONNX embedder (e.g., BGE-small). Keep dim (384) aligned with store.
- Reranker: implement `Reranker` using ONNX `bge-reranker-base` (cross-encoder), batching doc pairs for speed.

### ONNX toggle (build tag)
- Build with `-tags onnx` and set env:
  - `ONNX_EMBED_MODEL=./vectordb/models/bge-small-en-v1.5/model.onnx` (auto-detected if present)
  - `ONNX_EMBED_TOKENIZER=./vectordb/models/bge-small-en-v1.5/tokenizer.json` (auto-detected if present)
  - `ONNX_RERANK_MODEL=./models/bge-reranker-base.onnx`
  - `ONNX_RERANK_TOKENIZER=./models/tokenizer.json`
- Optional: `ONNX_EMBED_MAX_LEN`, `ONNX_RERANK_MAX_LEN`, `EMBED_DIM` (default 384).
- Without the tag, the code falls back to the hash embedder + simple cosine reranker.

### Fetch helper
```bash
./vectordb/fetch_bge_small.sh
# downloads model.onnx + tokenizer.json into vectordb/models/bge-small-en-v1.5/
```

## Runtime Notes
- Warm the model at startup with a dummy call.
- Ensure `VectorStore` capacity matches ingestion volume (`capacity * dim * 4 bytes`).
- If you need persistence, append vectors/metadata to a file (or mmap) alongside the flat buffer.
- Keep `DefaultTimeout` for the retrieval tool small (e.g., 200 ms) to enforce tail latency.

## Running
```bash
go run ./vectordb
# ensure the embedder is initialized and RetrievalTool uses it instead of the rand mock
```
