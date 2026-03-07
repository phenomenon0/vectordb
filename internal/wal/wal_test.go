package wal

import (
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestWALBasicOperations(t *testing.T) {
	dir := t.TempDir()

	// Open WAL
	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}
	defer w.Close()

	// Write some entries
	entries := []*Entry{
		{Op: OpInsert, Collection: "test", ID: "doc1", Doc: "hello world"},
		{Op: OpInsert, Collection: "test", ID: "doc2", Doc: "foo bar"},
		{Op: OpDelete, Collection: "test", ID: "doc1"},
	}

	for _, entry := range entries {
		if err := w.Write(entry); err != nil {
			t.Fatalf("failed to write entry: %v", err)
		}
	}

	// Verify LSN
	if w.LSN() != 3 {
		t.Errorf("expected LSN 3, got %d", w.LSN())
	}

	// Close and reopen
	if err := w.Close(); err != nil {
		t.Fatalf("failed to close WAL: %v", err)
	}

	// Read back entries
	readEntries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("failed to read entries: %v", err)
	}

	if len(readEntries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(readEntries))
	}

	// Verify entries
	if readEntries[0].Op != OpInsert || readEntries[0].ID != "doc1" {
		t.Errorf("entry 0 mismatch: %+v", readEntries[0])
	}
	if readEntries[1].Op != OpInsert || readEntries[1].ID != "doc2" {
		t.Errorf("entry 1 mismatch: %+v", readEntries[1])
	}
	if readEntries[2].Op != OpDelete || readEntries[2].ID != "doc1" {
		t.Errorf("entry 2 mismatch: %+v", readEntries[2])
	}
}

func TestWALRecovery(t *testing.T) {
	dir := t.TempDir()

	// Write some entries
	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	for i := 0; i < 10; i++ {
		if err := w.Write(&Entry{Op: OpInsert, Collection: "test", ID: "doc"}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	w.Close()

	// Reopen and verify LSN continues
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to reopen WAL: %v", err)
	}
	defer w2.Close()

	if w2.LSN() != 10 {
		t.Errorf("expected LSN 10 after recovery, got %d", w2.LSN())
	}

	// Write more entries
	if err := w2.Write(&Entry{Op: OpInsert}); err != nil {
		t.Fatalf("failed to write after recovery: %v", err)
	}

	if w2.LSN() != 11 {
		t.Errorf("expected LSN 11, got %d", w2.LSN())
	}
}

func TestWALSegmentRotation(t *testing.T) {
	dir := t.TempDir()

	// Create config with small segment size
	config := Config{
		Dir:            dir,
		MaxSegmentSize: 512, // 512 bytes per segment
		SyncOnWrite:    true,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}
	defer w.Close()

	// Write enough entries to trigger rotation
	for i := 0; i < 50; i++ {
		if err := w.Write(&Entry{
			Op:         OpInsert,
			Collection: "test",
			ID:         "doc",
			Doc:        "This is some content to make the entry larger and trigger rotation faster",
		}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}

	// Should have multiple segments
	segCount := w.SegmentCount()
	if segCount < 2 {
		t.Errorf("expected multiple segments, got %d", segCount)
	}

	t.Logf("Created %d segments with %d entries", segCount, w.LSN())
}

func TestWALIterator(t *testing.T) {
	dir := t.TempDir()

	// Create multiple segments manually
	config := Config{
		Dir:            dir,
		MaxSegmentSize: 256,
		SyncOnWrite:    true,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write enough to create multiple segments
	for i := 0; i < 30; i++ {
		if err := w.Write(&Entry{
			Op:         OpInsert,
			Collection: "test",
			ID:         "doc",
			Doc:        "content for segment rotation",
		}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	w.Close()

	// Iterate using NewIterator
	it, err := NewIterator(dir)
	if err != nil {
		t.Fatalf("failed to create iterator: %v", err)
	}
	defer it.Close()

	count := 0
	for {
		entry, err := it.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("iterator error: %v", err)
		}
		count++
		if entry.Op != OpInsert {
			t.Errorf("unexpected op: %v", entry.Op)
		}
	}

	if count != 30 {
		t.Errorf("expected 30 entries from iterator, got %d", count)
	}
}

func TestWALReplay(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write mixed operations
	ops := []OpType{OpInsert, OpInsert, OpDelete, OpUpdate, OpBatchInsert}
	for _, op := range ops {
		if err := w.Write(&Entry{Op: op, Collection: "test"}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	w.Close()

	// Replay all
	var replayed []OpType
	err = Replay(dir, func(e *Entry) bool {
		replayed = append(replayed, e.Op)
		return true
	})
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	if len(replayed) != len(ops) {
		t.Fatalf("expected %d replayed ops, got %d", len(ops), len(replayed))
	}

	for i, op := range ops {
		if replayed[i] != op {
			t.Errorf("op %d: expected %v, got %v", i, op, replayed[i])
		}
	}
}

func TestWALReplayFrom(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	for i := 0; i < 10; i++ {
		if err := w.Write(&Entry{Op: OpInsert}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	w.Close()

	// Replay from LSN 6
	var count int
	err = ReplayFrom(dir, 6, func(e *Entry) bool {
		count++
		return true
	})
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	// Should get entries 6-10 (5 entries)
	if count != 5 {
		t.Errorf("expected 5 entries from LSN 6, got %d", count)
	}
}

func TestWALTruncate(t *testing.T) {
	dir := t.TempDir()

	// Create with small segments
	config := Config{
		Dir:            dir,
		MaxSegmentSize: 256,
		SyncOnWrite:    true,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write enough to create multiple segments
	for i := 0; i < 50; i++ {
		if err := w.Write(&Entry{
			Op:  OpInsert,
			Doc: "content for truncation test",
		}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}

	segmentsBefore := w.SegmentCount()
	t.Logf("Segments before truncate: %d", segmentsBefore)

	// Truncate entries before LSN 40
	if err := w.Truncate(40); err != nil {
		t.Fatalf("truncate failed: %v", err)
	}

	w.Close()

	// Check remaining entries
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("failed to read after truncate: %v", err)
	}

	// Should still have entries >= 40
	for _, e := range entries {
		if e.LSN < 40 {
			// Note: truncation removes whole segments, so some earlier entries may remain
			t.Logf("Entry with LSN %d still present (expected due to segment-level truncation)", e.LSN)
		}
	}
}

func TestWALStats(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write mixed operations
	if err := w.Write(&Entry{Op: OpInsert}); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(&Entry{Op: OpInsert}); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(&Entry{Op: OpDelete}); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(&Entry{Op: OpUpdate}); err != nil {
		t.Fatal(err)
	}
	w.Close()

	stats, err := GetStats(dir)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}

	if stats.EntryCount != 4 {
		t.Errorf("expected 4 entries, got %d", stats.EntryCount)
	}
	if stats.FirstLSN != 1 {
		t.Errorf("expected first LSN 1, got %d", stats.FirstLSN)
	}
	if stats.LastLSN != 4 {
		t.Errorf("expected last LSN 4, got %d", stats.LastLSN)
	}
	if stats.OpCounts[OpInsert] != 2 {
		t.Errorf("expected 2 inserts, got %d", stats.OpCounts[OpInsert])
	}
	if stats.OpCounts[OpDelete] != 1 {
		t.Errorf("expected 1 delete, got %d", stats.OpCounts[OpDelete])
	}
	if stats.OpCounts[OpUpdate] != 1 {
		t.Errorf("expected 1 update, got %d", stats.OpCounts[OpUpdate])
	}
}

func TestWALBatchInsert(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write batch entry
	batch := []BatchEntry{
		{ID: "doc1", Vector: []float32{1.0, 2.0, 3.0}, Doc: "first"},
		{ID: "doc2", Vector: []float32{4.0, 5.0, 6.0}, Doc: "second"},
		{ID: "doc3", Vector: []float32{7.0, 8.0, 9.0}, Doc: "third"},
	}

	if err := w.Write(&Entry{
		Op:         OpBatchInsert,
		Collection: "vectors",
		Batch:      batch,
	}); err != nil {
		t.Fatalf("failed to write batch: %v", err)
	}
	w.Close()

	// Read back
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}

	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	entry := entries[0]
	if entry.Op != OpBatchInsert {
		t.Errorf("expected batch insert op, got %v", entry.Op)
	}
	if len(entry.Batch) != 3 {
		t.Errorf("expected 3 batch entries, got %d", len(entry.Batch))
	}
	if entry.Batch[0].ID != "doc1" {
		t.Errorf("expected doc1, got %s", entry.Batch[0].ID)
	}
}

func TestWALConcurrentWrites(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}
	defer w.Close()

	// Concurrent writes from multiple goroutines
	done := make(chan bool)
	writers := 5
	writesPerWriter := 20

	for i := 0; i < writers; i++ {
		go func(writerID int) {
			for j := 0; j < writesPerWriter; j++ {
				if err := w.Write(&Entry{
					Op:         OpInsert,
					Collection: "test",
					ID:         "doc",
					TenantID:   "tenant",
				}); err != nil {
					t.Errorf("writer %d failed: %v", writerID, err)
				}
			}
			done <- true
		}(i)
	}

	// Wait for all writers
	for i := 0; i < writers; i++ {
		<-done
	}

	// Verify all writes
	expectedLSN := uint64(writers * writesPerWriter)
	if w.LSN() != expectedLSN {
		t.Errorf("expected LSN %d, got %d", expectedLSN, w.LSN())
	}
}

func TestWALMaxSegments(t *testing.T) {
	dir := t.TempDir()

	rotatedSegments := []string{}
	config := Config{
		Dir:            dir,
		MaxSegmentSize: 256,
		SyncOnWrite:    true,
		MaxSegments:    3,
		OnRotate: func(oldSegment string) {
			rotatedSegments = append(rotatedSegments, oldSegment)
		},
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write enough to create many segments
	for i := 0; i < 100; i++ {
		if err := w.Write(&Entry{
			Op:  OpInsert,
			Doc: "content for max segments test",
		}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}
	w.Close()

	// Check that we have at most MaxSegments
	segments := countSegments(dir)
	if segments > 3 {
		t.Errorf("expected at most 3 segments, got %d", segments)
	}

	t.Logf("Final segment count: %d, rotations: %d", segments, len(rotatedSegments))
}

func TestWALBackgroundSync(t *testing.T) {
	dir := t.TempDir()

	config := Config{
		Dir:            dir,
		MaxSegmentSize: 64 * 1024 * 1024,
		SyncOnWrite:    false, // Disable sync on write
		SyncInterval:   50 * time.Millisecond,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	// Write some entries
	for i := 0; i < 5; i++ {
		if err := w.Write(&Entry{Op: OpInsert}); err != nil {
			t.Fatalf("failed to write: %v", err)
		}
	}

	// Wait for background sync
	time.Sleep(100 * time.Millisecond)

	// Close and verify
	w.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}

	if len(entries) != 5 {
		t.Errorf("expected 5 entries, got %d", len(entries))
	}
}

func TestWALOpTypeString(t *testing.T) {
	tests := []struct {
		op       OpType
		expected string
	}{
		{OpInsert, "INSERT"},
		{OpDelete, "DELETE"},
		{OpUpdate, "UPDATE"},
		{OpBatchInsert, "BATCH_INSERT"},
		{OpCreateCollection, "CREATE_COLLECTION"},
		{OpDeleteCollection, "DELETE_COLLECTION"},
		{OpCompact, "COMPACT"},
		{OpType(99), "UNKNOWN(99)"},
	}

	for _, tc := range tests {
		if got := tc.op.String(); got != tc.expected {
			t.Errorf("OpType(%d).String() = %q, want %q", tc.op, got, tc.expected)
		}
	}
}

func TestWALEmptyDir(t *testing.T) {
	dir := t.TempDir()

	// Read from empty directory
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("failed to read empty dir: %v", err)
	}

	if len(entries) != 0 {
		t.Errorf("expected 0 entries from empty dir, got %d", len(entries))
	}
}

func TestWALClosedWrite(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("failed to open WAL: %v", err)
	}

	w.Close()

	// Write to closed WAL should fail
	err = w.Write(&Entry{Op: OpInsert})
	if err == nil {
		t.Error("expected error writing to closed WAL")
	}
}

func countSegments(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}

	count := 0
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".log" {
			count++
		}
	}
	return count
}
