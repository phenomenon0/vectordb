package cluster

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	cowrie "github.com/Neumenon/cowrie/go"
	"github.com/phenomenon0/vectordb/internal/cowrieutil"
)

// decodeSnapshotAuto reads a snapshot file and auto-detects the format.
// Supports: cowrie+zstd (new), gzip+JSON (legacy), raw JSON (legacy uncompressed).
func decodeSnapshotAuto(file *os.File) (*snapshotFileData, error) {
	// Read first 2 bytes to detect format
	header := make([]byte, 2)
	if _, err := file.Read(header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	file.Seek(0, 0)

	// Gzip magic: 0x1f 0x8b → legacy gzip+JSON snapshot
	if header[0] == 0x1f && header[1] == 0x8b {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()

		raw, err := io.ReadAll(gzReader)
		if err != nil {
			return nil, fmt.Errorf("failed to read gzip data: %w", err)
		}

		if isJSONSnapshot(raw) {
			var data snapshotFileData
			if err := json.Unmarshal(raw, &data); err != nil {
				return nil, fmt.Errorf("failed to decode JSON snapshot: %w", err)
			}
			return &data, nil
		}
		// Gzip-wrapped cowrie (unlikely but handle it)
		return decodeSnapshotCowrie(raw)
	}

	// Read all bytes for format detection
	raw, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read snapshot: %w", err)
	}

	// JSON: starts with '{'
	if isJSONSnapshot(raw) {
		var data snapshotFileData
		if err := json.Unmarshal(raw, &data); err != nil {
			return nil, fmt.Errorf("failed to decode JSON snapshot: %w", err)
		}
		return &data, nil
	}

	// Cowrie binary (raw or framed+zstd)
	return decodeSnapshotCowrie(raw)
}

// encodeSnapshotCowrie serializes snapshot data using cowrie binary format.
// Vectors are encoded as a native float32 tensor (~60% smaller than JSON).
// The output is cowrie+zstd compressed for maximum size reduction.
func encodeSnapshotCowrie(data *snapshotFileData) ([]byte, error) {
	obj := cowrie.Object()

	// Metadata sub-object
	if data.Metadata != nil {
		meta := cowrie.Object()
		meta.Set("id", cowrie.String(data.Metadata.ID))
		meta.Set("created_at", cowrie.String(data.Metadata.CreatedAt.Format(time.RFC3339Nano)))
		meta.Set("wal_sequence", cowrie.Int64(int64(data.Metadata.WALSequence)))
		meta.Set("vector_count", cowrie.Int64(int64(data.Metadata.VectorCount)))
		meta.Set("dimension", cowrie.Int64(int64(data.Metadata.Dimension)))
		meta.Set("collections", cowrie.Int64(int64(data.Metadata.Collections)))
		meta.Set("size_bytes", cowrie.Int64(data.Metadata.SizeBytes))
		if data.Metadata.Checksum != "" {
			meta.Set("checksum", cowrie.String(data.Metadata.Checksum))
		}
		meta.Set("compressed", cowrie.Bool(data.Metadata.Compressed))
		meta.Set("version", cowrie.Int64(int64(data.Metadata.Version)))
		obj.Set("metadata", meta)
	}

	// Vector data — the big win: tensor encoding vs JSON float array
	if len(data.Data) > 0 {
		obj.Set("data", cowrieutil.EncodeFloat32Tensor(data.Data))
	}

	// String arrays
	if len(data.Docs) > 0 {
		obj.Set("docs", cowrieutil.EncodeStringArray(data.Docs))
	}
	if len(data.IDs) > 0 {
		obj.Set("ids", cowrieutil.EncodeStringArray(data.IDs))
	}
	if len(data.Seqs) > 0 {
		obj.Set("seqs", cowrieutil.EncodeUint64Array(data.Seqs))
	}

	// Maps
	if len(data.Meta) > 0 {
		obj.Set("meta", cowrieutil.EncodeStringMapMap(data.Meta))
	}
	if len(data.NumMeta) > 0 {
		obj.Set("num_meta", cowrieutil.EncodeFloat64MapMap(data.NumMeta))
	}
	if len(data.TimeMeta) > 0 {
		obj.Set("time_meta", cowrieutil.EncodeTimeMapMap(data.TimeMeta))
	}
	if len(data.Coll) > 0 {
		obj.Set("coll", cowrieutil.EncodeStringMapUint64(data.Coll))
	}
	if len(data.TenantID) > 0 {
		obj.Set("tenant_id", cowrieutil.EncodeStringMapUint64(data.TenantID))
	}
	if len(data.LexTF) > 0 {
		obj.Set("lex_tf", cowrieutil.EncodeIntMapMap(data.LexTF))
	}
	if len(data.DocLen) > 0 {
		obj.Set("doc_len", cowrieutil.EncodeIntMapUint64(data.DocLen))
	}
	if len(data.DF) > 0 {
		obj.Set("df", cowrieutil.EncodeStringIntMap(data.DF))
	}
	if data.SumDocL > 0 {
		obj.Set("sum_doc_l", cowrie.Int64(int64(data.SumDocL)))
	}

	// Use zstd framed encoding — snapshots are large and latency-tolerant
	return cowrie.EncodeFramed(obj, cowrie.CompressionZstd)
}

// decodeSnapshotCowrie deserializes snapshot data from cowrie binary format.
func decodeSnapshotCowrie(raw []byte) (*snapshotFileData, error) {
	// Try framed (zstd) first, fall back to raw cowrie
	obj, err := cowrie.DecodeFramed(raw)
	if err != nil {
		obj, err = cowrie.Decode(raw)
		if err != nil {
			return nil, fmt.Errorf("cowrie decode failed: %w", err)
		}
	}

	data := &snapshotFileData{}

	// Metadata
	if v := obj.Get("metadata"); v != nil && v.Type() == cowrie.TypeObject {
		m := &StoreSnapshot{}
		if iv := v.Get("id"); iv != nil {
			m.ID = cowrieutil.SafeString(iv)
		}
		if iv := v.Get("created_at"); iv != nil {
			if t, err := time.Parse(time.RFC3339Nano, cowrieutil.SafeString(iv)); err == nil {
				m.CreatedAt = t
			}
		}
		if iv := v.Get("wal_sequence"); iv != nil {
			m.WALSequence = cowrieutil.SafeUint64(iv)
		}
		if iv := v.Get("vector_count"); iv != nil {
			m.VectorCount = int(cowrieutil.SafeInt64(iv))
		}
		if iv := v.Get("dimension"); iv != nil {
			m.Dimension = int(cowrieutil.SafeInt64(iv))
		}
		if iv := v.Get("collections"); iv != nil {
			m.Collections = int(cowrieutil.SafeInt64(iv))
		}
		if iv := v.Get("size_bytes"); iv != nil {
			m.SizeBytes = cowrieutil.SafeInt64(iv)
		}
		if iv := v.Get("checksum"); iv != nil {
			m.Checksum = cowrieutil.SafeString(iv)
		}
		if iv := v.Get("compressed"); iv != nil {
			m.Compressed = cowrieutil.SafeBool(iv)
		}
		if iv := v.Get("version"); iv != nil {
			m.Version = int(cowrieutil.SafeInt64(iv))
		}
		data.Metadata = m
	}

	// Vector data
	if v := obj.Get("data"); v != nil {
		data.Data = cowrieutil.DecodeFloat32Tensor(v)
	}

	// String arrays
	if v := obj.Get("docs"); v != nil {
		data.Docs = cowrieutil.DecodeStringArray(v)
	}
	if v := obj.Get("ids"); v != nil {
		data.IDs = cowrieutil.DecodeStringArray(v)
	}
	if v := obj.Get("seqs"); v != nil {
		data.Seqs = cowrieutil.DecodeUint64Array(v)
	}

	// Maps
	if v := obj.Get("meta"); v != nil {
		data.Meta = cowrieutil.DecodeStringMapMap(v)
	}
	if v := obj.Get("num_meta"); v != nil {
		data.NumMeta = cowrieutil.DecodeFloat64MapMap(v)
	}
	if v := obj.Get("time_meta"); v != nil {
		data.TimeMeta = cowrieutil.DecodeTimeMapMap(v)
	}
	if v := obj.Get("coll"); v != nil {
		data.Coll = cowrieutil.DecodeStringMapUint64(v)
	}
	if v := obj.Get("tenant_id"); v != nil {
		data.TenantID = cowrieutil.DecodeStringMapUint64(v)
	}
	if v := obj.Get("lex_tf"); v != nil {
		data.LexTF = cowrieutil.DecodeIntMapMap(v)
	}
	if v := obj.Get("doc_len"); v != nil {
		data.DocLen = cowrieutil.DecodeIntMapUint64(v)
	}
	if v := obj.Get("df"); v != nil {
		data.DF = cowrieutil.DecodeStringIntMap(v)
	}
	if v := obj.Get("sum_doc_l"); v != nil {
		data.SumDocL = int(cowrieutil.SafeInt64(v))
	}

	return data, nil
}

// isJSONSnapshot returns true if the raw data starts with '{', indicating JSON format.
func isJSONSnapshot(data []byte) bool {
	return len(data) > 0 && data[0] == '{'
}
