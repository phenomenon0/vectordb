# VectorDB Index Abstraction Layer

**Status:** Phase 1 Complete ✅
**Created:** 2025-12-09
**Part of:** VectorDB Enhancement Plan (12-month roadmap)

## Overview

This package provides a pluggable index abstraction layer for VectorDB, enabling support for multiple vector index types while maintaining backward compatibility and Cowrie optimization.

## Design Philosophy

- **Simple**: Minimal interface with core operations
- **Pluggable**: Easy to swap index implementations
- **Efficient**: Supports both in-memory and disk-backed indexes
- **Observable**: Returns metrics and statistics
- **Cowrie-Compatible**: Export/Import uses JSON (upgradeable to Cowrie)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  VectorStore                        │
│  (manages collections, multi-tenancy, persistence)  │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌─────────────────────────────────────────────────────┐
│             Index Interface                         │
│  Add() Search() Delete() Stats() Export() Import()  │
└───────┬─────────────────┬───────────────────────────┘
        │                 │
        │                 │ implements
        ▼                 ▼
┌───────────────┐  ┌──────────────┐   ┌──────────────┐
│  HNSWIndex    │  │  IVFIndex    │   │  FlatIndex   │
│ (implemented) │  │  (planned)   │   │  (planned)   │
└───────────────┘  └──────────────┘   └──────────────┘
```

## Index Interface

All index types implement this core interface:

```go
type Index interface {
    Name() string
    Add(ctx context.Context, id uint64, vector []float32) error
    Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error)
    Delete(ctx context.Context, id uint64) error
    Stats() IndexStats
    Export() ([]byte, error)
    Import(data []byte) error
}
```

## Implemented Index Types

### HNSW (Hierarchical Navigable Small World)

**Status:** ✅ Implemented
**Best for:** General-purpose approximate nearest neighbor search
**Complexity:** O(log n) search time
**Memory:** ~(M * 2 * 8 * count) + (count * dim * 4) bytes

**Configuration:**
```go
idx, err := index.Create("hnsw", 384, map[string]interface{}{
    "m":         16,   // Connections per node (5-48)
    "ml":        0.25, // Level multiplier
    "ef_search": 64,   // Search beam width (higher = better recall)
})
```

**Features:**
- ✅ Incremental construction (no retraining)
- ✅ High recall (>95% with proper parameters)
- ✅ Thread-safe concurrent reads
- ✅ Tombstone deletion with compaction
- ✅ Export/Import for persistence

**Limitations:**
- Approximate results (not exact)
- Memory-resident (no disk backing yet)
- Delete requires compaction for memory reclamation

## Planned Index Types

### IVF (Inverted File)
**Status:** ⏳ Planned (Phase 3)
**Best for:** 10M-100M vectors with recall trade-off
**Complexity:** O(nlist + nprobe * avg_cluster_size)

### FLAT (Brute Force)
**Status:** ⏳ Planned (Phase 3)
**Best for:** <100K vectors, exact results required
**Complexity:** O(n) exact search

### DiskANN
**Status:** ⏳ Planned (Phase 3)
**Best for:** 100M+ vectors with memory constraints
**Complexity:** Hybrid memory + disk

## Usage Examples

### Creating an Index

```go
import "agentscope/vectordb/index"

// Using factory
idx, err := index.Create("hnsw", 384, map[string]interface{}{
    "m": 16,
    "ef_search": 64,
})

// Or directly
idx, err := index.NewHNSWIndex(384, map[string]interface{}{
    "m": 16,
})
```

### Adding Vectors

```go
ctx := context.Background()

vec := make([]float32, 384)
// ... populate vector ...

if err := idx.Add(ctx, 1, vec); err != nil {
    log.Fatal(err)
}
```

### Searching

```go
// Basic search
results, err := idx.Search(ctx, query, 10, index.DefaultSearchParams{})

// With HNSW-specific parameters
results, err := idx.Search(ctx, query, 10, index.HNSWSearchParams{
    EfSearch: 128, // Higher = better recall, slower
})

for _, r := range results {
    fmt.Printf("ID: %d, Distance: %.4f, Score: %.4f\n",
        r.ID, r.Distance, r.Score)
}
```

### Statistics

```go
stats := idx.Stats()
fmt.Printf("Index: %s\n", stats.Name)
fmt.Printf("Dimension: %d\n", stats.Dim)
fmt.Printf("Total vectors: %d\n", stats.Count)
fmt.Printf("Active vectors: %d\n", stats.Active)
fmt.Printf("Deleted vectors: %d\n", stats.Deleted)
fmt.Printf("Memory used: %d bytes\n", stats.MemoryUsed)
```

### Persistence

```go
// Export to JSON
data, err := idx.Export()
if err != nil {
    log.Fatal(err)
}

// Save to file
os.WriteFile("index.json", data, 0644)

// Import from JSON
data, err := os.ReadFile("index.json")
if err != nil {
    log.Fatal(err)
}

if err := idx.Import(data); err != nil {
    log.Fatal(err)
}
```

### Compaction (HNSW-specific)

```go
// After many deletions, compact to reclaim memory
hnswIdx := idx.(*index.HNSWIndex)
removed, err := hnswIdx.Compact()
fmt.Printf("Compacted: removed %d deleted vectors\n", removed)
```

## Factory Pattern

Register custom index types:

```go
import "agentscope/vectordb/index"

func init() {
    index.Register("myindex", func(dim int, config map[string]interface{}) (index.Index, error) {
        return NewMyIndex(dim, config)
    })
}

// Now available via factory
idx, err := index.Create("myindex", 384, config)
```

## Testing

```bash
cd vectordb/index
go test -v

# Run specific test
go test -v -run TestHNSWBasicOperations

# With coverage
go test -cover
```

## Performance Characteristics

### HNSW Index

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Add | O(log n) | Incremental, no rebuild needed |
| Search | O(log n) | Approximate (recall >95%) |
| Delete | O(1) | Tombstone (lazy) |
| Compact | O(n log n) | Rebuilds graph excluding deleted |
| Export | O(n) | Serializes all vectors + metadata |
| Import | O(n log n) | Rebuilds graph from vectors |

### Memory Usage

```
Total = Graph + Vectors + Mappings
Graph ≈ M * 2 * 8 * count bytes
Vectors ≈ count * dim * 4 bytes
Mappings ≈ count * 16 bytes

Example: 1M vectors, 384 dim, M=16
= 16 * 2 * 8 * 1M + 1M * 384 * 4 + 1M * 16
= 256 MB + 1.5 GB + 16 MB
≈ 1.77 GB
```

## Cowrie Compatibility

The index abstraction is designed to work seamlessly with Cowrie:

### Current State (Phase 1)
- Export/Import use JSON format (human-readable, debuggable)
- JSON is compatible with Cowrie (can be upgraded)
- Vectors stored as `[]float32` arrays

### Future Enhancement (Phase 2+)
When Cowrie tensors are needed:

```go
// Export will use Cowrie with tensor encoding
type exportFormatCowrie struct {
    Version int
    Dim     int
    Config  map[string]interface{}
    Vectors *ucodec.TensorV1  // 48% smaller than JSON
    Deleted []uint64
}
```

**Benefits:**
- 48% size reduction for float32 arrays
- Zero-copy mmap support
- Streaming large indexes

**Upgrade Path:**
1. Add Cowrie export format alongside JSON
2. Auto-detect format on Import
3. Deprecate JSON export (with migration period)

## Integration with VectorStore

Future integration (Phase 1 completion):

```go
// VectorStore will use Index interface
type VectorStore struct {
    sync.RWMutex
    indexes map[string]index.Index  // Collection -> Index mapping
    // ... other fields ...
}

// Create collection with specific index type
func (vs *VectorStore) CreateCollection(name string, indexType string, dim int) error {
    idx, err := index.Create(indexType, dim, config)
    if err != nil {
        return err
    }
    vs.indexes[name] = idx
    return nil
}
```

## Files in This Package

| File | Lines | Purpose |
|------|-------|---------|
| `interface.go` | 235 | Core Index interface, Result, SearchParams |
| `factory.go` | 175 | Factory pattern, registration, config helpers |
| `hnsw.go` | 428 | HNSW implementation wrapping github.com/coder/hnsw |
| `hnsw_test.go` | 280 | Comprehensive test suite for HNSW |
| `README.md` | (this) | Documentation |

**Total:** ~1,118 lines

## Related Documentation

- [VectorDB Enhancement Plan](../../.claude/plans/gleaming-mixing-book.md) - 12-month roadmap
- [VectorDB README](../README.md) - Main VectorDB documentation
- [Cowrie Codec](../../cowrie/README.md) - Cowrie format documentation

## Contributing

When adding new index types:

1. **Implement Index interface**
2. **Register with factory** in `init()`
3. **Add comprehensive tests** (cover all interface methods)
4. **Document configuration** parameters
5. **Add performance benchmarks**
6. **Update this README**

Example template:

```go
package index

type MyIndex struct {
    // ... fields ...
}

func NewMyIndex(dim int, config map[string]interface{}) (Index, error) {
    // ... implementation ...
}

func (m *MyIndex) Name() string { return "MyIndex" }
func (m *MyIndex) Add(ctx context.Context, id uint64, vector []float32) error { ... }
func (m *MyIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) { ... }
func (m *MyIndex) Delete(ctx context.Context, id uint64) error { ... }
func (m *MyIndex) Stats() IndexStats { ... }
func (m *MyIndex) Export() ([]byte, error) { ... }
func (m *MyIndex) Import(data []byte) error { ... }

func init() {
    Register("myindex", func(dim int, config map[string]interface{}) (Index, error) {
        return NewMyIndex(dim, config)
    })
}
```

## Roadmap

### Phase 1 (Complete) ✅
- [x] Index interface design
- [x] Factory pattern
- [x] HNSW implementation
- [x] Comprehensive tests
- [x] Documentation

### Phase 2 (Next)
- [ ] Integrate with VectorStore
- [ ] Collection management
- [ ] Cowrie export format
- [ ] Migration utilities

### Phase 3 (Future)
- [ ] IVF index implementation
- [ ] FLAT index implementation
- [ ] DiskANN index implementation
- [ ] GPU-accelerated indexes

---

**Status**: Foundation complete. Ready for VectorStore integration.
