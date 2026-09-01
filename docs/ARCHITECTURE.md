# DeepData architecture

Prose arrives with slice S3 (plan 7.2 J3); until then this file exists for the block below, which `scripts/check_docs_contract.py` reads as the non-goal truth for rules R8 and R11.

<!-- non-goals -->
DiskANN | \bdiskann\b
IVF | \bivf(?:[_-]?(?:flat|pq|hnsw|sq))?\b|\binverted[- ]file\b
PQ/binary/scalar quantization | \b(?:product|binary|scalar|int8|uint8|fp16|float16)[- ]quantiz\w*|\bPQ\b|\bquantiz(?:ation|ed|er|e)\b
replication/cluster/shard/failover | \b(?:replication|clustering|shards?|sharding|sharded|failover|follower restore|multi[- ]node|high[- ]availability|distributed mode)\b|(?<!one )(?<!single )(?<!\d )\breplicas?\b|(?<!kubernetes )(?<!kind )(?<!k8s )\bclusters?\b(?![- ]level)
GraphRAG/graph reranking | \bgraph[- ]?rag\b|\bgraph[- ](?:rerank\w*|boost\w*)\b|\bknowledge[- ]graph\b|\bpagerank\b
Tauri/desktop app | \btauri\b|\bdesktop (?:app|application|installer|package|build|client)s?\b
web UI | \bweb[- ]?ui\b|\bbrowser[- ]based (?:ui|dashboard|console)\b|\balpine\.?js\b|\bvite\b
CUDA/GPU | \bcuda\b|\bgpus?\b|\bgpu-accelerated\b|\bnvidia\b
built-in TLS/encryption-at-rest | \bencryption[- ]at[- ]rest\b|\b(?:built[- ]in|native|server[- ]side)\s+(?:m?tls|encryption)\b|\btls (?:cert\w*|key|listener|config\w*)\b
server-managed embeddings (remove when CTL-02 lands) | \bserver[- ]managed embedd\w*|\bembedding providers?\b|\b(?:ollama|openai|gemini|voyage|jina|cohere|mistral)[- ]embedd\w*|\bONNX\b|\bbge[- ]small\b|/api/embed\b
<!-- /non-goals -->
