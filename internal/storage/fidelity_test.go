package storage

import (
	"bytes"
	"reflect"
	"testing"
	"time"
)

// TestEveryRegisteredFormatRoundTripsEveryPayloadField characterizes the
// minimum fidelity required from every supported snapshot codec. A production
// snapshot must not silently discard fields needed to reconstruct ownership,
// vector encodings, pagination, or collection indexes after restart.
func TestEveryRegisteredFormatRoundTripsEveryPayloadField(t *testing.T) {
	savedAt := time.Date(2026, time.July, 17, 12, 34, 56, 789, time.UTC)
	indexedAt := time.Date(2025, time.December, 1, 2, 3, 4, 5, time.UTC)

	want := &Payload{
		FormatVersion: 2,
		Dim:           3,
		Data:          []float32{1, 2, 3, 4, 5, 6},
		VectorType:    2,
		VectorData: map[uint64][]byte{
			11: {0x01, 0x02, 0x03},
			22: {0xfe, 0xff},
		},
		Docs: []string{"first document", "second document"},
		IDs:  []string{"doc-1", "doc-2"},
		Seqs: []uint64{17, 23},
		Meta: map[uint64]map[string]string{
			11: {"kind": "primary", "region": "us"},
			22: {"kind": "secondary"},
		},
		Deleted: map[uint64]bool{22: true},
		Coll: map[uint64]string{
			11: "alpha",
			22: "beta",
		},
		TenantID: map[uint64]string{
			11: "tenant-a",
			22: "tenant-b",
		},
		Next:         23,
		NextSeq:      31,
		WALHighWater: 47,
		Count:        2,
		HNSW:         []byte("legacy-hnsw-payload"),
		Indexes: map[string][]byte{
			"alpha": []byte(`{"version":2,"type":"hnsw"}`),
			"beta":  []byte(`{"version":2,"type":"flat"}`),
		},
		IndexTypes:     map[string]string{"alpha": "hnsw", "beta": "flat"},
		IndexDims:      map[string]int{"alpha": 3, "beta": 3},
		IndexChecksums: map[string]string{"alpha": "sha256:alpha", "beta": "sha256:beta"},
		Checksum:       "sha256:characterization",
		LastSaved:      savedAt,
		LexTF: map[uint64]map[string]int{
			11: {"first": 1, "document": 1},
			22: {"second": 1, "document": 1},
		},
		DocLen:  map[uint64]int{11: 2, 22: 2},
		DF:      map[string]int{"document": 2, "first": 1, "second": 1},
		SumDocL: 4,
		NumMeta: map[uint64]map[string]float64{
			11: {"score": 0.75},
			22: {"score": 1.25},
		},
		TimeMeta: map[uint64]map[string]time.Time{
			11: {"indexed_at": indexedAt},
			22: {"indexed_at": indexedAt.Add(time.Hour)},
		},
	}

	for _, formatName := range []string{"gob", "cowrie", "cowrie-zstd", "cowrie-delta-zstd"} {
		format := Get(formatName)
		if format == nil {
			t.Fatalf("format %q is not registered", formatName)
		}
		t.Run(formatName, func(t *testing.T) {
			var encoded bytes.Buffer
			if err := format.Save(&encoded, want); err != nil {
				t.Fatalf("save failed: %v", err)
			}
			got, err := format.Load(bytes.NewReader(encoded.Bytes()))
			if err != nil {
				t.Fatalf("load failed: %v", err)
			}

			fields := []struct {
				name string
				got  interface{}
				want interface{}
			}{
				{"FormatVersion", got.FormatVersion, want.FormatVersion},
				{"Dim", got.Dim, want.Dim},
				{"Data", got.Data, want.Data},
				{"VectorType", got.VectorType, want.VectorType},
				{"VectorData", got.VectorData, want.VectorData},
				{"Docs", got.Docs, want.Docs},
				{"IDs", got.IDs, want.IDs},
				{"Seqs", got.Seqs, want.Seqs},
				{"Meta", got.Meta, want.Meta},
				{"Deleted", got.Deleted, want.Deleted},
				{"Coll", got.Coll, want.Coll},
				{"TenantID", got.TenantID, want.TenantID},
				{"Next", got.Next, want.Next},
				{"NextSeq", got.NextSeq, want.NextSeq},
				{"WALHighWater", got.WALHighWater, want.WALHighWater},
				{"Count", got.Count, want.Count},
				{"HNSW", got.HNSW, want.HNSW},
				{"Indexes", got.Indexes, want.Indexes},
				{"IndexTypes", got.IndexTypes, want.IndexTypes},
				{"IndexDims", got.IndexDims, want.IndexDims},
				{"IndexChecksums", got.IndexChecksums, want.IndexChecksums},
				{"Checksum", got.Checksum, want.Checksum},
				{"LastSaved", got.LastSaved, want.LastSaved},
				{"LexTF", got.LexTF, want.LexTF},
				{"DocLen", got.DocLen, want.DocLen},
				{"DF", got.DF, want.DF},
				{"SumDocL", got.SumDocL, want.SumDocL},
				{"NumMeta", got.NumMeta, want.NumMeta},
				{"TimeMeta", got.TimeMeta, want.TimeMeta},
			}

			for _, field := range fields {
				if !reflect.DeepEqual(field.got, field.want) {
					t.Errorf("%s round-trip mismatch: got %#v, want %#v", field.name, field.got, field.want)
				}
			}
		})
	}
}
