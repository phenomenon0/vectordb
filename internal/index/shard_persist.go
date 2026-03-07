package index

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	shard "github.com/Neumenon/shard/go/shard"
)

// ShardRoleVectorIndex is the Shard v2 role for DeepData vector indexes.
const ShardRoleVectorIndex shard.ShardRole = 0x10

// ShardIndexMeta is stored as JSON in the "meta" entry.
type ShardIndexMeta struct {
	IndexType string `json:"index_type"` // "hnsw", "ivf", "flat", "diskann", "pq", "sparse", "binary"
	Version   int    `json:"version"`
	Dim       int    `json:"dim"`
	Count     int    `json:"count"`
	CreatedAt string `json:"created_at"`
}

// ExportToShard serializes an index into a Shard v2 container file.
// The index's Export() output is stored as a zstd-compressed entry with CRC32C integrity.
func ExportToShard(idx Index, path string) error {
	data, err := idx.Export()
	if err != nil {
		return fmt.Errorf("export index: %w", err)
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	w, err := shard.NewShardV2Writer(path, ShardRoleVectorIndex)
	if err != nil {
		return fmt.Errorf("create shard writer: %w", err)
	}
	defer w.Close()

	w.SetAlignment(shard.Align64)
	w.SetCompression(shard.CompressZstd)

	stats := idx.Stats()
	meta := ShardIndexMeta{
		IndexType: idx.Name(),
		Version:   1,
		Dim:       stats.Dim,
		Count:     stats.Count,
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}

	if err := w.WriteEntryTyped("meta", metaJSON, shard.ContentTypeJSON); err != nil {
		return fmt.Errorf("write meta entry: %w", err)
	}

	if err := w.WriteEntryCompressed("index_data", data); err != nil {
		return fmt.Errorf("write index data entry: %w", err)
	}

	return w.Close()
}

// ImportFromShard deserializes an index from a Shard v2 container file.
// Returns a new index populated with the stored data.
func ImportFromShard(path string, factory func(indexType string, dim int) (Index, error)) (Index, error) {
	r, err := shard.OpenShardV2(path)
	if err != nil {
		return nil, fmt.Errorf("open shard: %w", err)
	}
	defer r.Close()

	// Read and parse metadata
	metaIdx := r.Lookup("meta")
	if metaIdx < 0 {
		return nil, fmt.Errorf("shard missing 'meta' entry")
	}
	metaData, err := r.ReadEntry(metaIdx)
	if err != nil {
		return nil, fmt.Errorf("read meta: %w", err)
	}

	var meta ShardIndexMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return nil, fmt.Errorf("parse meta: %w", err)
	}

	// Create the appropriate index type
	idx, err := factory(meta.IndexType, meta.Dim)
	if err != nil {
		return nil, fmt.Errorf("create index %q: %w", meta.IndexType, err)
	}

	// Read index data (decompressed + checksum-verified automatically)
	dataIdx := r.Lookup("index_data")
	if dataIdx < 0 {
		return nil, fmt.Errorf("shard missing 'index_data' entry")
	}
	data, err := r.ReadEntry(dataIdx)
	if err != nil {
		return nil, fmt.Errorf("read index data: %w", err)
	}

	if err := idx.Import(data); err != nil {
		return nil, fmt.Errorf("import index data: %w", err)
	}

	return idx, nil
}

// ReadShardMeta reads only the metadata from a Shard v2 index file without loading the full index.
func ReadShardMeta(path string) (*ShardIndexMeta, error) {
	r, err := shard.OpenShardV2(path)
	if err != nil {
		return nil, fmt.Errorf("open shard: %w", err)
	}
	defer r.Close()

	metaIdx := r.Lookup("meta")
	if metaIdx < 0 {
		return nil, fmt.Errorf("shard missing 'meta' entry")
	}
	metaData, err := r.ReadEntry(metaIdx)
	if err != nil {
		return nil, fmt.Errorf("read meta: %w", err)
	}

	var meta ShardIndexMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return nil, fmt.Errorf("parse meta: %w", err)
	}
	return &meta, nil
}
