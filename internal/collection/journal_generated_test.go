package collection

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// These tests generate a small journal through the real DurableStore writer,
// then corrupt bytes at computed frame offsets and reopen. They complement the
// hand-built-frame parser tests in journal_test.go by proving the end-to-end
// startup contract on real mutation frames.
//
// Already covered elsewhere and deliberately not duplicated here:
//   - crash/reopen state parity without a checkpoint:
//     TestDurableStoreReplayRetainsJournalUntilExplicitCheckpoint
//   - snapshot format compatibility (v1 legacy migration, v2/v3 preservation,
//     checksum fail-closed): TestUnifiedCollectionSnapshotLegacyMigration,
//     TestDurableStorePreservesV2AndV3SnapshotState,
//     TestUnifiedCollectionSnapshotRejectsChecksumCorruption
//   - two-pass recovery and bounded retained payloads:
//     TestCollectionJournalValidationAndReplayReuseOnePayloadBufferAndBindArtifacts,
//     TestDurableStoreRejectsMalformedMutationBeforeReplay

type generatedJournalFrame struct{ start, end int }

// generatedJournalFixture is one create-collection plus three inserts
// (LSNs 1..4) abandoned without a checkpoint, so every acknowledged record
// lives only in the current journal and must survive reopen.
type generatedJournalFixture struct {
	base     string
	journal  string
	snapshot string
	original []byte
	frames   []generatedJournalFrame
}

func newGeneratedJournalFixture(t *testing.T) *generatedJournalFixture {
	t.Helper()
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		doc := durableTestDocument(float32(i))
		if err := store.Tenants().AddDocument(ctx, "tenant", "docs", &doc); err != nil {
			t.Fatal(err)
		}
	}
	if got := store.Metadata().AppliedLSN; got != 4 {
		t.Fatalf("applied LSN = %d, want 4", got)
	}
	abandonDurableStoreForTest(t, store)

	f := &generatedJournalFixture{base: base, journal: base + ".journal", snapshot: collectionSnapshotPath(base)}
	f.original = f.readFile(t, f.journal)
	for off := 0; off < len(f.original); {
		payloadLen := int(binary.BigEndian.Uint32(f.original[off+collectionJournalPayloadLenOffset:]))
		end := off + int(collectionJournalHeaderSize) + payloadLen
		f.frames = append(f.frames, generatedJournalFrame{start: off, end: end})
		off = end
	}
	if len(f.frames) != 4 || f.frames[3].end != len(f.original) {
		t.Fatalf("generated journal frames = %+v for %d bytes, want 4 exact frames", f.frames, len(f.original))
	}
	return f
}

func (f *generatedJournalFixture) readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func (f *generatedJournalFixture) writeJournal(t *testing.T, data []byte) {
	t.Helper()
	if err := os.WriteFile(f.journal, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// assertState checks the recovered high-water mark and that exactly documents
// 1..wantDocs are present, which is how acknowledged-write loss shows up.
func (f *generatedJournalFixture) assertState(t *testing.T, store *DurableStore, wantLSN uint64, wantDocs int) {
	t.Helper()
	if got := store.Metadata().AppliedLSN; got != wantLSN {
		t.Fatalf("recovered LSN = %d, want %d", got, wantLSN)
	}
	if got := durableTestCollectionInfo(t, store, "tenant", "docs").DocCount; got != wantDocs {
		t.Fatalf("recovered document count = %d, want %d", got, wantDocs)
	}
	for id := uint64(1); id <= 3; id++ {
		_, ok := durableTestStoredDocument(t, store, "tenant", "docs", id)
		if want := id <= uint64(wantDocs); ok != want {
			t.Fatalf("document %d present = %v, want %v", id, ok, want)
		}
	}
}

// TestGeneratedJournalCorruptRecordsFailClosed guards the invariant that a
// corrupt or reordered acknowledged record can never be silently repaired,
// skipped, or partially applied: startup must refuse, must leave the journal
// and snapshot bytes untouched so an operator can restore them, and once the
// original bytes are restored every acknowledged record must still replay.
// Each case corrupts frame 2 (LSN 2, the first insert) so that a violation
// would surface as lost documents behind a "successful" open.
func TestGeneratedJournalCorruptRecordsFailClosed(t *testing.T) {
	headerSize := int(collectionJournalHeaderSize)
	flip := func(offset func(generatedJournalFrame) int) func([]byte, []generatedJournalFrame) []byte {
		return func(data []byte, frames []generatedJournalFrame) []byte {
			data[offset(frames[1])] ^= 0xff
			return data
		}
	}
	setLength := func(length func([]byte, generatedJournalFrame) uint32) func([]byte, []generatedJournalFrame) []byte {
		return func(data []byte, frames []generatedJournalFrame) []byte {
			binary.BigEndian.PutUint32(data[frames[1].start+collectionJournalPayloadLenOffset:], length(data, frames[1]))
			return data
		}
	}

	tests := []struct {
		name    string
		mutate  func(data []byte, frames []generatedJournalFrame) []byte
		wantErr string
	}{
		{
			name:    "payload byte flip",
			mutate:  flip(func(fr generatedJournalFrame) int { return fr.start + headerSize }),
			wantErr: "checksum mismatch",
		},
		{
			name:    "checksum byte flip",
			mutate:  flip(func(fr generatedJournalFrame) int { return fr.start + collectionJournalChecksumOffset }),
			wantErr: "checksum mismatch",
		},
		{
			name:    "length header zero",
			mutate:  setLength(func([]byte, generatedJournalFrame) uint32 { return 0 }),
			wantErr: "checksum mismatch",
		},
		{
			name:    "length header above maximum",
			mutate:  setLength(func([]byte, generatedJournalFrame) uint32 { return collectionJournalMaxPayload + 1 }),
			wantErr: "maximum",
		},
		{
			// A complete, acknowledged frame whose length field grew past EOF
			// looks like a torn tail to the framing scanner. Repairing it
			// would truncate this and every later acknowledged record.
			name: "length header past EOF under maximum",
			mutate: setLength(func(data []byte, fr generatedJournalFrame) uint32 {
				return uint32(len(data) - fr.start - headerSize + 1)
			}),
			wantErr: "corrupt payload length",
		},
		{
			name: "truncated interior record",
			mutate: func(data []byte, frames []generatedJournalFrame) []byte {
				mid := frames[1].start + headerSize + (frames[1].end-frames[1].start-headerSize)/2
				return append(data[:mid], data[mid+8:]...)
			},
			wantErr: "checksum mismatch",
		},
		{
			name: "interior record removed",
			mutate: func(data []byte, frames []generatedJournalFrame) []byte {
				return append(data[:frames[1].start], data[frames[1].end:]...)
			},
			wantErr: "LSN gap",
		},
		{
			name: "interior record duplicated",
			mutate: func(data []byte, frames []generatedJournalFrame) []byte {
				dup := append([]byte(nil), data[frames[1].start:frames[1].end]...)
				return append(data[:frames[1].end], append(dup, data[frames[1].end:]...)...)
			},
			wantErr: "duplicates or regresses",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newGeneratedJournalFixture(t)
			corrupt := tc.mutate(append([]byte(nil), f.original...), f.frames)
			if bytes.Equal(corrupt, f.original) {
				t.Fatal("mutation did not change the journal")
			}
			f.writeJournal(t, corrupt)
			snapshotBefore := f.readFile(t, f.snapshot)

			store, err := OpenDurableStore(f.base, f.base)
			if err == nil {
				_ = store.Abort()
				t.Fatal("corrupt journal unexpectedly opened")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("corrupt journal error = %v, want %q", err, tc.wantErr)
			}
			if got := f.readFile(t, f.journal); !bytes.Equal(got, corrupt) {
				t.Fatal("failed open rewrote the corrupt journal")
			}
			if got := f.readFile(t, f.snapshot); !bytes.Equal(got, snapshotBefore) {
				t.Fatal("failed open rewrote the snapshot")
			}

			// Restoring the original bytes must recover every acknowledged
			// record: the failed open applied and persisted nothing.
			f.writeJournal(t, f.original)
			restored, err := OpenDurableStore(f.base, f.base)
			if err != nil {
				t.Fatalf("reopen after restoring original journal: %v", err)
			}
			f.assertState(t, restored, 4, 3)
			if err := restored.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestGeneratedJournalTerminalPartialTailRepairsExactlyOnce is the one
// permitted repair: a torn final frame was never acknowledged, so truncating
// it back to the last complete frame loses nothing. The repair must happen
// exactly once and a second open must find nothing further to remove;
// otherwise a repair loop could eat acknowledged frames one restart at a time.
func TestGeneratedJournalTerminalPartialTailRepairsExactlyOnce(t *testing.T) {
	f := newGeneratedJournalFixture(t)
	last := f.frames[3]
	headerSize := int(collectionJournalHeaderSize)
	cut := last.start + headerSize + (last.end-last.start-headerSize)/2
	f.writeJournal(t, f.original[:cut])

	store, err := OpenDurableStore(f.base, f.base)
	if err != nil {
		t.Fatalf("open with torn terminal frame: %v", err)
	}
	f.assertState(t, store, 3, 2)
	if got := f.readFile(t, f.journal); !bytes.Equal(got, f.original[:last.start]) {
		t.Fatalf("repaired journal is %d bytes, want the %d-byte complete prefix", len(got), last.start)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(f.base, f.base)
	if err != nil {
		t.Fatalf("reopen after repair: %v", err)
	}
	f.assertState(t, reopened, 3, 2)
	if got := f.readFile(t, f.journal); !bytes.Equal(got, f.original[:last.start]) {
		t.Fatal("second open changed the already-repaired journal")
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestGeneratedJournalEmptySparseVectorReplays guards a record the live HTTP
// path writes today: the handler turns a client's "indices": [] into a typed
// *sparse.SparseVector, cloneDocumentValue copies its slices with
// append([]uint32(nil), ...), which is nil, and the journal frame therefore
// carries "indices":null. Replay decodes that as a nil interface and must read
// it as the empty sparse vector it was; failing closed there lets one document
// with no sparse terms make the whole store unopenable after a restart (RCV-05
// hit this at LSN 480 of the preserved 4.71 GB journal).
func TestGeneratedJournalEmptySparseVectorReplays(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	schema := CollectionSchema{Name: "docs", Fields: []VectorField{
		{Name: "embedding", Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeFLAT}},
		{Name: "text", Type: VectorTypeSparse, Dim: 8, Index: IndexConfig{Type: IndexTypeInverted}},
	}}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", schema); err != nil {
		t.Fatal(err)
	}
	// One term, then no terms at all, both as the typed vectors the HTTP layer hands the engine.
	for _, sparseVec := range []*sparse.SparseVector{
		mustSparseVector(t, []uint32{1}, []float32{1}, 8),
		mustSparseVector(t, nil, nil, 8),
	} {
		doc := Document{Vectors: map[string]Vector{"embedding": Vector{Dense: []float32{1, 0, 0, 0}}, "text": Vector{Sparse: sparseVec}}}
		if err := store.Tenants().AddDocument(ctx, "tenant", "docs", &doc); err != nil {
			t.Fatal(err)
		}
	}
	abandonDurableStoreForTest(t, store)

	journal, err := os.ReadFile(base + ".journal")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(journal, []byte(`"indices":null`)) {
		t.Fatal("writer no longer journals an empty sparse vector as null; inject the legacy encoding so this compatibility test keeps exercising it")
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen refused a journal holding an empty sparse vector: %v", err)
	}
	if got := durableTestCollectionInfo(t, reopened, "tenant", "docs").DocCount; got != 2 {
		t.Fatalf("recovered document count = %d, want 2", got)
	}
	doc, ok := durableTestStoredDocument(t, reopened, "tenant", "docs", 2)
	if !ok {
		t.Fatal("document 2 (empty sparse) missing after replay")
	}
	if sv := doc.Vectors["text"].Sparse; sv == nil || len(sv.Indices) != 0 || sv.Dim != 8 {
		t.Fatalf("document 2 sparse field = %#v, want empty *sparse.SparseVector over dim 8", doc.Vectors["text"])
	}
	abandonDurableStoreForTest(t, reopened)
}

func mustSparseVector(t *testing.T, indices []uint32, values []float32, dim int) *sparse.SparseVector {
	t.Helper()
	v, err := sparse.NewSparseVector(indices, values, dim)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
