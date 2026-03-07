package index

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	shard "github.com/Neumenon/shard/go/shard"
)

// Shard v2 columnar entry names for index persistence.
// Instead of one monolithic "index_data" blob, we split into separate entries
// so callers can load only what they need (e.g., just config, just vectors).
const (
	ColMeta    = "col/meta"    // JSON: ShardIndexMeta
	ColConfig  = "col/config"  // JSON: index-specific config (m, ef_search, etc.)
	ColVectors = "col/vectors" // Binary: packed [id:uint64 | dim*float32] rows
	ColDeleted = "col/deleted" // Binary: packed uint64 deleted IDs
	ColGraph   = "col/graph"   // Opaque: index-specific graph structure (future)
)

// ColumnExportable is implemented by indexes that support columnar export.
// This is optional — indexes that don't implement it fall back to monolithic Export().
type ColumnExportable interface {
	// ExportConfig returns index configuration as JSON-marshalable map.
	ExportConfig() map[string]interface{}
	// ExportVectors returns all vectors as (id, vector) pairs.
	ExportVectors() []VectorEntry
	// ExportDeleted returns IDs of soft-deleted vectors.
	ExportDeleted() []uint64
}

// VectorEntry is a single vector with its ID, used for columnar export/import.
type VectorEntry struct {
	ID     uint64
	Vector []float32
}

// ExportToShardColumnar serializes an index into a Shard v2 container with
// separate columnar entries for config, vectors, and deleted IDs.
// This allows selective loading — e.g., read just metadata without touching vectors.
func ExportToShardColumnar(idx Index, path string) error {
	ce, ok := idx.(ColumnExportable)
	if !ok {
		// Fall back to monolithic export
		return ExportToShard(idx, path)
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

	// 1. Write metadata
	meta := ShardIndexMeta{
		IndexType: idx.Name(),
		Version:   2, // v2 = columnar layout
		Dim:       stats.Dim,
		Count:     stats.Count,
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}
	if err := w.WriteEntryTyped(ColMeta, metaJSON, shard.ContentTypeJSON); err != nil {
		return fmt.Errorf("write meta: %w", err)
	}

	// 2. Write config
	cfg := ce.ExportConfig()
	cfgJSON, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	if err := w.WriteEntryTyped(ColConfig, cfgJSON, shard.ContentTypeJSON); err != nil {
		return fmt.Errorf("write config: %w", err)
	}

	// 3. Write vectors as packed binary: [uint64 id][dim * float32]
	vectors := ce.ExportVectors()
	if len(vectors) > 0 {
		dim := stats.Dim
		rowSize := 8 + dim*4 // uint64 ID + dim float32s
		buf := make([]byte, len(vectors)*rowSize)
		for i, v := range vectors {
			off := i * rowSize
			binary.LittleEndian.PutUint64(buf[off:], v.ID)
			for j := 0; j < dim && j < len(v.Vector); j++ {
				binary.LittleEndian.PutUint32(buf[off+8+j*4:], math.Float32bits(v.Vector[j]))
			}
		}
		if err := w.WriteEntryCompressed(ColVectors, buf); err != nil {
			return fmt.Errorf("write vectors: %w", err)
		}
	}

	// 4. Write deleted IDs as packed uint64s
	deleted := ce.ExportDeleted()
	if len(deleted) > 0 {
		buf := make([]byte, len(deleted)*8)
		for i, id := range deleted {
			binary.LittleEndian.PutUint64(buf[i*8:], id)
		}
		if err := w.WriteEntryCompressed(ColDeleted, buf); err != nil {
			return fmt.Errorf("write deleted: %w", err)
		}
	}

	return w.Close()
}

// ColumnReader provides selective access to individual columns stored in a
// Shard v2 index file. Open once, read only what you need, close.
type ColumnReader struct {
	reader *shard.ShardV2Reader
	meta   *ShardIndexMeta
}

// OpenColumnReader opens a Shard v2 file for selective column access.
// Works with both monolithic (v1) and columnar (v2) shard layouts.
func OpenColumnReader(path string) (*ColumnReader, error) {
	r, err := shard.OpenShardV2(path)
	if err != nil {
		return nil, fmt.Errorf("open shard: %w", err)
	}

	cr := &ColumnReader{reader: r}

	// Try columnar meta first, fall back to monolithic
	metaKey := ColMeta
	idx := r.Lookup(metaKey)
	if idx < 0 {
		metaKey = "meta"
		idx = r.Lookup(metaKey)
	}
	if idx < 0 {
		r.Close()
		return nil, fmt.Errorf("shard missing metadata entry")
	}

	metaData, err := r.ReadEntry(idx)
	if err != nil {
		r.Close()
		return nil, fmt.Errorf("read meta: %w", err)
	}

	var meta ShardIndexMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		r.Close()
		return nil, fmt.Errorf("parse meta: %w", err)
	}
	cr.meta = &meta

	return cr, nil
}

// Meta returns the index metadata without loading any data.
func (cr *ColumnReader) Meta() *ShardIndexMeta {
	return cr.meta
}

// IsColumnar returns true if the shard uses columnar layout (v2+).
func (cr *ColumnReader) IsColumnar() bool {
	return cr.meta.Version >= 2
}

// HasColumn checks if a specific column exists in the shard.
func (cr *ColumnReader) HasColumn(name string) bool {
	return cr.reader.Lookup(name) >= 0
}

// Columns returns the list of all entry names in the shard.
func (cr *ColumnReader) Columns() []string {
	return cr.reader.EntryNames()
}

// ReadConfig returns the index configuration without loading vectors.
func (cr *ColumnReader) ReadConfig() (map[string]interface{}, error) {
	idx := cr.reader.Lookup(ColConfig)
	if idx < 0 {
		return nil, fmt.Errorf("no config column")
	}
	data, err := cr.reader.ReadEntry(idx)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg map[string]interface{}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return cfg, nil
}

// ReadVectors returns all vectors without loading config or deleted IDs.
func (cr *ColumnReader) ReadVectors() ([]VectorEntry, error) {
	idx := cr.reader.Lookup(ColVectors)
	if idx < 0 {
		return nil, fmt.Errorf("no vectors column")
	}
	data, err := cr.reader.ReadEntry(idx)
	if err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}
	return decodeVectorColumn(data, cr.meta.Dim)
}

// ReadDeleted returns soft-deleted IDs without loading vectors.
func (cr *ColumnReader) ReadDeleted() ([]uint64, error) {
	idx := cr.reader.Lookup(ColDeleted)
	if idx < 0 {
		return nil, nil // No deleted column = no deletions
	}
	data, err := cr.reader.ReadEntry(idx)
	if err != nil {
		return nil, fmt.Errorf("read deleted: %w", err)
	}
	return decodeDeletedColumn(data), nil
}

// ReadRaw reads a raw entry by name. Useful for custom columns or graph data.
func (cr *ColumnReader) ReadRaw(name string) ([]byte, error) {
	idx := cr.reader.Lookup(name)
	if idx < 0 {
		return nil, fmt.Errorf("entry %q not found", name)
	}
	return cr.reader.ReadEntry(idx)
}

// EntrySize returns the compressed disk size of an entry without reading it.
// Returns -1 if the entry doesn't exist.
func (cr *ColumnReader) EntrySize(name string) int64 {
	idx := cr.reader.Lookup(name)
	if idx < 0 {
		return -1
	}
	info := cr.reader.GetEntryInfo(idx)
	if info == nil {
		return -1
	}
	return int64(info.DiskSize)
}

// Close releases the underlying reader.
func (cr *ColumnReader) Close() error {
	return cr.reader.Close()
}

// ImportFromShardColumnar loads an index from a columnar Shard v2 file.
// Falls back to monolithic import if the shard uses v1 layout.
func ImportFromShardColumnar(path string, factory func(indexType string, dim int) (Index, error)) (Index, error) {
	cr, err := OpenColumnReader(path)
	if err != nil {
		return nil, err
	}
	defer cr.Close()

	if !cr.IsColumnar() {
		// Fall back to monolithic import
		cr.Close()
		return ImportFromShard(path, factory)
	}

	meta := cr.Meta()
	idx, err := factory(meta.IndexType, meta.Dim)
	if err != nil {
		return nil, fmt.Errorf("create index %q: %w", meta.IndexType, err)
	}

	// Read config
	cfg, err := cr.ReadConfig()
	if err != nil {
		return nil, err
	}

	// Read vectors
	vectors, err := cr.ReadVectors()
	if err != nil {
		return nil, err
	}

	// Read deleted
	deleted, err := cr.ReadDeleted()
	if err != nil {
		return nil, err
	}

	// Build the monolithic JSON the existing Import() expects
	type vectorJSON struct {
		ID     uint64    `json:"id"`
		Vector []float32 `json:"vector"`
	}
	vecs := make([]vectorJSON, len(vectors))
	for i, v := range vectors {
		vecs[i] = vectorJSON{ID: v.ID, Vector: v.Vector}
	}

	importData := struct {
		Version int                    `json:"version"`
		Dim     int                    `json:"dim"`
		Config  map[string]interface{} `json:"config"`
		Vectors []vectorJSON           `json:"vectors"`
		Deleted []uint64               `json:"deleted"`
	}{
		Version: 1,
		Dim:     meta.Dim,
		Config:  cfg,
		Vectors: vecs,
		Deleted: deleted,
	}

	data, err := json.Marshal(importData)
	if err != nil {
		return nil, fmt.Errorf("marshal import data: %w", err)
	}

	if err := idx.Import(data); err != nil {
		return nil, fmt.Errorf("import: %w", err)
	}
	return idx, nil
}

// decodeVectorColumn unpacks binary vector data: [uint64 id][dim * float32] per row.
func decodeVectorColumn(data []byte, dim int) ([]VectorEntry, error) {
	rowSize := 8 + dim*4
	if rowSize == 0 {
		return nil, fmt.Errorf("invalid dimension: %d", dim)
	}
	if len(data)%rowSize != 0 {
		return nil, fmt.Errorf("vector data size %d not divisible by row size %d", len(data), rowSize)
	}
	count := len(data) / rowSize
	entries := make([]VectorEntry, count)
	for i := 0; i < count; i++ {
		off := i * rowSize
		id := binary.LittleEndian.Uint64(data[off:])
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[off+8+j*4:]))
		}
		entries[i] = VectorEntry{ID: id, Vector: vec}
	}
	return entries, nil
}

// decodeDeletedColumn unpacks binary deleted IDs: packed uint64s.
func decodeDeletedColumn(data []byte) []uint64 {
	count := len(data) / 8
	ids := make([]uint64, count)
	for i := 0; i < count; i++ {
		ids[i] = binary.LittleEndian.Uint64(data[i*8:])
	}
	return ids
}
