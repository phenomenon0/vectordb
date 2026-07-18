# DeepData Lessons

- When a user asks to commit a wide fix set, run the older package-level short suite before calling the branch verified; targeted regression tests are not enough.
- Listener-based Go tests should use an explicit helper instead of raw `httptest.NewServer` so restricted environments skip cleanly rather than panicking on socket creation.
- Benchmark checkpoints must upsert by a complete cell identity, write atomically, clear stale failures on success, and validate the intended matrix before destructive invalidation.
- Cross-database recall requires explicit primary keys that match ground-truth row IDs; generated database IDs are not insertion offsets.
- For Milvus, insert and flush before creating HNSW, wait for index completion, verify indexed row counts, and ensure the effective `ef` is at least the requested top-k.
- Resume orchestration should derive infrastructure from genuinely pending cells so completed work never starts unrelated services or collides with occupied ports.
