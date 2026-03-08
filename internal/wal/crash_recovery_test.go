package wal

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// TestCrashRecovery_BasicWriteThenRecover writes entries, closes the WAL,
// reopens it, and verifies all entries are recovered with correct content.
func TestCrashRecovery_BasicWriteThenRecover(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	// Write entries with distinct, verifiable content
	type expected struct {
		op         OpType
		collection string
		id         string
		doc        string
	}
	want := []expected{
		{OpInsert, "users", "u1", "alice"},
		{OpInsert, "users", "u2", "bob"},
		{OpUpdate, "users", "u1", "alice-updated"},
		{OpDelete, "users", "u2", ""},
		{OpInsert, "products", "p1", "widget"},
	}

	for _, e := range want {
		if err := w.Write(&Entry{
			Op:         e.op,
			Collection: e.collection,
			ID:         e.id,
			Doc:        e.doc,
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	// Simulate crash: close without any special cleanup
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopen — recovery should restore LSN
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	if w2.LSN() != uint64(len(want)) {
		t.Errorf("recovered LSN = %d, want %d", w2.LSN(), len(want))
	}

	// Read back all entries and verify content
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(entries) != len(want) {
		t.Fatalf("recovered %d entries, want %d", len(entries), len(want))
	}

	for i, e := range entries {
		if e.Op != want[i].op {
			t.Errorf("entry %d: op = %v, want %v", i, e.Op, want[i].op)
		}
		if e.Collection != want[i].collection {
			t.Errorf("entry %d: collection = %q, want %q", i, e.Collection, want[i].collection)
		}
		if e.ID != want[i].id {
			t.Errorf("entry %d: id = %q, want %q", i, e.ID, want[i].id)
		}
		if e.Doc != want[i].doc {
			t.Errorf("entry %d: doc = %q, want %q", i, e.Doc, want[i].doc)
		}
		if e.LSN != uint64(i+1) {
			t.Errorf("entry %d: LSN = %d, want %d", i, e.LSN, i+1)
		}
	}
}

// TestCrashRecovery_ConcurrentWriteRecovery runs multiple goroutines writing
// concurrently, then closes (simulating a crash) and verifies no data is lost.
func TestCrashRecovery_ConcurrentWriteRecovery(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	const writers = 8
	const writesPerWriter = 50
	const totalWrites = writers * writesPerWriter

	var wg sync.WaitGroup
	var writeErrors atomic.Int64

	for g := 0; g < writers; g++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			for j := 0; j < writesPerWriter; j++ {
				if err := w.Write(&Entry{
					Op:         OpInsert,
					Collection: "concurrent",
					ID:         fmt.Sprintf("w%d-d%d", writerID, j),
					Doc:        fmt.Sprintf("writer-%d-doc-%d", writerID, j),
				}); err != nil {
					writeErrors.Add(1)
				}
			}
		}(g)
	}

	wg.Wait()

	if n := writeErrors.Load(); n != 0 {
		t.Fatalf("%d write errors during concurrent writes", n)
	}

	finalLSN := w.LSN()
	if finalLSN != totalWrites {
		t.Fatalf("LSN after concurrent writes = %d, want %d", finalLSN, totalWrites)
	}

	// Crash
	w.Close()

	// Recover
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	if w2.LSN() != totalWrites {
		t.Errorf("recovered LSN = %d, want %d", w2.LSN(), totalWrites)
	}

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(entries) != totalWrites {
		t.Errorf("recovered %d entries, want %d", len(entries), totalWrites)
	}

	// Verify all LSNs are present (1..totalWrites), no gaps
	seen := make(map[uint64]bool, totalWrites)
	for _, e := range entries {
		if seen[e.LSN] {
			t.Errorf("duplicate LSN %d", e.LSN)
		}
		seen[e.LSN] = true
	}
	for lsn := uint64(1); lsn <= totalWrites; lsn++ {
		if !seen[lsn] {
			t.Errorf("missing LSN %d", lsn)
		}
	}
}

// TestCrashRecovery_PartialWrite simulates a crash mid-write by truncating the
// WAL file so the last entry is incomplete. Recovery should return all complete
// entries and skip the partial one.
func TestCrashRecovery_PartialWrite(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	const totalEntries = 20
	for i := 0; i < totalEntries; i++ {
		if err := w.Write(&Entry{
			Op:         OpInsert,
			Collection: "partial",
			ID:         fmt.Sprintf("doc-%d", i),
			Doc:        "some content here for the partial write test scenario",
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	w.Close()

	// Find the segment file and truncate the last few bytes to simulate
	// a crash that interrupted the final write.
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	if len(segFiles) == 0 {
		t.Fatal("no segment files found")
	}

	lastSeg := segFiles[len(segFiles)-1]
	info, err := os.Stat(lastSeg)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	// Chop 15 bytes off the end — enough to corrupt the last entry
	truncSize := info.Size() - 15
	if truncSize < 0 {
		truncSize = 0
	}
	if err := os.Truncate(lastSeg, truncSize); err != nil {
		t.Fatalf("truncate: %v", err)
	}

	// Recover — should not panic, should return at least totalEntries-1
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen after partial write: %v", err)
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		// ReadAll may return partial results + error; that's acceptable
		t.Logf("ReadAll returned error (expected): %v", err)
	}

	if len(entries) < totalEntries-1 {
		t.Errorf("recovered %d entries, want at least %d", len(entries), totalEntries-1)
	}
	if len(entries) >= totalEntries {
		t.Errorf("recovered %d entries — truncation should have lost at least one", len(entries))
	}

	t.Logf("recovered %d / %d entries after partial write", len(entries), totalEntries)

	// Verify recovered entries are valid and in order
	for i, e := range entries {
		if e.LSN != uint64(i+1) {
			t.Errorf("entry %d: LSN = %d, want %d", i, e.LSN, i+1)
		}
	}
}

// TestCrashRecovery_PartialHeader simulates a crash that wrote only part of
// the 8-byte header (length + CRC). Recovery should skip the incomplete header.
func TestCrashRecovery_PartialHeader(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	for i := 0; i < 5; i++ {
		if err := w.Write(&Entry{
			Op: OpInsert,
			ID: fmt.Sprintf("doc-%d", i),
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	w.Close()

	// Append 4 garbage bytes (partial header) to the segment
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	lastSeg := segFiles[len(segFiles)-1]
	f, err := os.OpenFile(lastSeg, os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		t.Fatalf("open for append: %v", err)
	}
	f.Write([]byte{0xDE, 0xAD, 0xBE, 0xEF}) // partial header
	f.Close()

	// Recovery should return the 5 valid entries
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Logf("ReadAll error (expected for partial header): %v", err)
	}
	if len(entries) != 5 {
		t.Errorf("recovered %d entries, want 5", len(entries))
	}
}

// TestCrashRecovery_LargeBatch writes 10K+ entries, recovers, and verifies
// count and content integrity.
func TestCrashRecovery_LargeBatch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large batch test in short mode")
	}

	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	const total = 10_000

	for i := 0; i < total; i++ {
		if err := w.Write(&Entry{
			Op:         OpInsert,
			Collection: "large",
			ID:         fmt.Sprintf("doc-%05d", i),
			Vector:     []float32{float32(i), float32(i + 1), float32(i + 2)},
			Doc:        fmt.Sprintf("document number %d", i),
		}); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}
	w.Close()

	// Recover
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	if w2.LSN() != total {
		t.Errorf("recovered LSN = %d, want %d", w2.LSN(), total)
	}

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(entries) != total {
		t.Fatalf("recovered %d entries, want %d", len(entries), total)
	}

	// Spot-check content
	for _, idx := range []int{0, 999, 4999, 9999} {
		e := entries[idx]
		wantID := fmt.Sprintf("doc-%05d", idx)
		if e.ID != wantID {
			t.Errorf("entry %d: id = %q, want %q", idx, e.ID, wantID)
		}
		if e.Vector[0] != float32(idx) {
			t.Errorf("entry %d: vector[0] = %f, want %f", idx, e.Vector[0], float32(idx))
		}
	}
}

// TestCrashRecovery_AfterDelete writes inserts and deletes, recovers, and
// verifies both insert and delete entries are preserved in the WAL.
func TestCrashRecovery_AfterDelete(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	// Insert 10 docs
	for i := 0; i < 10; i++ {
		if err := w.Write(&Entry{
			Op:         OpInsert,
			Collection: "col",
			ID:         fmt.Sprintf("doc-%d", i),
			Doc:        fmt.Sprintf("content-%d", i),
		}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	// Delete even-numbered docs
	for i := 0; i < 10; i += 2 {
		if err := w.Write(&Entry{
			Op:         OpDelete,
			Collection: "col",
			ID:         fmt.Sprintf("doc-%d", i),
		}); err != nil {
			t.Fatalf("delete: %v", err)
		}
	}

	w.Close()

	// Recover
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	const totalOps = 15 // 10 inserts + 5 deletes
	if w2.LSN() != totalOps {
		t.Errorf("recovered LSN = %d, want %d", w2.LSN(), totalOps)
	}

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(entries) != totalOps {
		t.Fatalf("recovered %d entries, want %d", len(entries), totalOps)
	}

	// Replay and build final state
	alive := make(map[string]bool)
	for _, e := range entries {
		switch e.Op {
		case OpInsert:
			alive[e.ID] = true
		case OpDelete:
			delete(alive, e.ID)
		}
	}

	// Even docs (0,2,4,6,8) should be deleted, odd docs (1,3,5,7,9) remain
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("doc-%d", i)
		if i%2 == 0 {
			if alive[id] {
				t.Errorf("doc %s should be deleted but is alive", id)
			}
		} else {
			if !alive[id] {
				t.Errorf("doc %s should be alive but is missing", id)
			}
		}
	}
}

// TestCrashRecovery_RepeatedCycles performs multiple write-crash-recover
// cycles and verifies cumulative state is preserved across all cycles.
func TestCrashRecovery_RepeatedCycles(t *testing.T) {
	dir := t.TempDir()

	const cycles = 5
	const writesPerCycle = 20
	var cumulativeLSN uint64

	for cycle := 0; cycle < cycles; cycle++ {
		w, err := Open(DefaultConfig(dir))
		if err != nil {
			t.Fatalf("cycle %d open: %v", cycle, err)
		}

		// Verify LSN was recovered correctly
		if w.LSN() != cumulativeLSN {
			t.Fatalf("cycle %d: recovered LSN = %d, want %d", cycle, w.LSN(), cumulativeLSN)
		}

		for j := 0; j < writesPerCycle; j++ {
			if err := w.Write(&Entry{
				Op:         OpInsert,
				Collection: "cycles",
				ID:         fmt.Sprintf("c%d-d%d", cycle, j),
			}); err != nil {
				t.Fatalf("cycle %d write %d: %v", cycle, j, err)
			}
		}

		cumulativeLSN += writesPerCycle

		// Crash (close)
		w.Close()
	}

	// Final verification
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("final ReadAll: %v", err)
	}

	totalExpected := cycles * writesPerCycle
	if len(entries) != totalExpected {
		t.Errorf("total entries = %d, want %d", len(entries), totalExpected)
	}

	// Verify LSNs are monotonically increasing 1..totalExpected
	for i, e := range entries {
		if e.LSN != uint64(i+1) {
			t.Errorf("entry %d: LSN = %d, want %d", i, e.LSN, i+1)
		}
	}

	// Verify entries from each cycle are present
	cycleCount := make(map[int]int)
	for _, e := range entries {
		var c, d int
		fmt.Sscanf(e.ID, "c%d-d%d", &c, &d)
		cycleCount[c]++
	}
	for c := 0; c < cycles; c++ {
		if cycleCount[c] != writesPerCycle {
			t.Errorf("cycle %d: %d entries, want %d", c, cycleCount[c], writesPerCycle)
		}
	}
}

// TestCrashRecovery_CorruptedEntry injects garbage bytes in the middle of
// the WAL. Recovery should return entries before the corruption without panicking.
func TestCrashRecovery_CorruptedEntry(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	for i := 0; i < 10; i++ {
		if err := w.Write(&Entry{
			Op: OpInsert,
			ID: fmt.Sprintf("doc-%d", i),
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	w.Close()

	// Read the file and inject corruption after entry 5
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	lastSeg := segFiles[len(segFiles)-1]
	data, err := os.ReadFile(lastSeg)
	if err != nil {
		t.Fatalf("readfile: %v", err)
	}

	// Find the byte offset after the 5th entry by scanning
	offset := 0
	for i := 0; i < 5; i++ {
		if offset+8 > len(data) {
			t.Fatalf("segment too short to find entry %d", i)
		}
		entryLen := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 8 + int(entryLen) // header + data
	}

	// Replace bytes at offset with garbage (corrupt the 6th entry's header)
	garbage := []byte{0xFF, 0xFF, 0xFF, 0x7F, 0xBA, 0xAD, 0xCA, 0xFE}
	copy(data[offset:offset+8], garbage)

	if err := os.WriteFile(lastSeg, data, 0644); err != nil {
		t.Fatalf("writefile: %v", err)
	}

	// Recovery should not panic
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		// Some implementations may refuse to open; that's acceptable
		t.Logf("open after corruption returned error (acceptable): %v", err)
		return
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Logf("ReadAll after corruption returned error (expected): %v", err)
	}

	// We should recover at least the first 5 valid entries
	if len(entries) < 5 {
		t.Errorf("recovered %d entries, want at least 5 before corruption", len(entries))
	}

	t.Logf("recovered %d / 10 entries with mid-file corruption", len(entries))
}

// TestCrashRecovery_CorruptedCRC writes valid-length entries but with a bad
// CRC. The reader should detect and reject them.
func TestCrashRecovery_CorruptedCRC(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	for i := 0; i < 3; i++ {
		if err := w.Write(&Entry{
			Op: OpInsert,
			ID: fmt.Sprintf("doc-%d", i),
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	w.Close()

	// Corrupt the CRC of the 2nd entry
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	lastSeg := segFiles[len(segFiles)-1]
	data, err := os.ReadFile(lastSeg)
	if err != nil {
		t.Fatalf("readfile: %v", err)
	}

	// Skip past entry 0 to find entry 1
	entry0Len := binary.LittleEndian.Uint32(data[0:4])
	entry1Offset := 8 + int(entry0Len)
	// Flip CRC bytes (offset+4 to offset+8)
	data[entry1Offset+4] ^= 0xFF
	data[entry1Offset+5] ^= 0xFF

	if err := os.WriteFile(lastSeg, data, 0644); err != nil {
		t.Fatalf("writefile: %v", err)
	}

	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Logf("open after CRC corruption: %v", err)
		return
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Logf("ReadAll with CRC error (expected): %v", err)
	}

	// Should get at least entry 0 before the CRC mismatch stops reading
	if len(entries) < 1 {
		t.Errorf("recovered %d entries, want at least 1", len(entries))
	}
	// Should NOT get all 3 since entry 1 has bad CRC
	if len(entries) == 3 {
		t.Error("recovered all 3 entries despite CRC corruption — CRC check may be broken")
	}

	t.Logf("recovered %d / 3 entries with CRC corruption on entry 1", len(entries))
}

// TestCrashRecovery_EmptySegmentFile creates an empty segment file and
// verifies recovery handles it gracefully.
func TestCrashRecovery_EmptySegmentFile(t *testing.T) {
	dir := t.TempDir()

	// Write some entries
	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for i := 0; i < 5; i++ {
		w.Write(&Entry{Op: OpInsert, ID: fmt.Sprintf("doc-%d", i)})
	}
	w.Close()

	// Create an empty segment file with a higher number (simulates crash
	// right after segment rotation, before any data was written)
	emptySegPath := filepath.Join(dir, "wal-00000002.log")
	f, err := os.Create(emptySegPath)
	if err != nil {
		t.Fatalf("create empty seg: %v", err)
	}
	f.Close()

	// Recovery should handle this fine
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(entries) != 5 {
		t.Errorf("recovered %d entries, want 5", len(entries))
	}
}

// TestCrashRecovery_GarbageFile puts a completely garbage file in the WAL
// directory and verifies the system handles it without panicking.
func TestCrashRecovery_GarbageFile(t *testing.T) {
	dir := t.TempDir()

	// Write valid entries first
	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for i := 0; i < 5; i++ {
		w.Write(&Entry{Op: OpInsert, ID: fmt.Sprintf("doc-%d", i)})
	}
	w.Close()

	// Create a segment file full of random garbage
	garbagePath := filepath.Join(dir, "wal-00000099.log")
	rng := rand.New(rand.NewSource(42))
	garbage := make([]byte, 1024)
	rng.Read(garbage)
	if err := os.WriteFile(garbagePath, garbage, 0644); err != nil {
		t.Fatalf("write garbage: %v", err)
	}

	// Recovery should not panic. It may error but must not crash.
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Logf("open with garbage segment: %v (acceptable)", err)
		// Even if Open errors, we should still be able to read the valid entries
		entries, _ := ReadAll(dir)
		if len(entries) < 5 {
			t.Errorf("recovered %d valid entries, want at least 5", len(entries))
		}
		return
	}
	defer w2.Close()

	// The iterator skips corrupted segments, so we should still get our entries
	entries, _ := ReadAll(dir)
	t.Logf("recovered %d entries with garbage segment present", len(entries))
	if len(entries) < 5 {
		t.Errorf("recovered %d entries, want at least 5", len(entries))
	}
}

// TestCrashRecovery_SegmentRotationMidCrash writes enough entries to trigger
// segment rotation, then simulates a crash by truncating the newest segment.
// Recovery should preserve all entries from completed segments.
func TestCrashRecovery_SegmentRotationMidCrash(t *testing.T) {
	dir := t.TempDir()

	config := Config{
		Dir:            dir,
		MaxSegmentSize: 512, // small segments for fast rotation
		SyncOnWrite:    true,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	// Write enough to create multiple segments
	const total = 50
	for i := 0; i < total; i++ {
		if err := w.Write(&Entry{
			Op:  OpInsert,
			ID:  fmt.Sprintf("doc-%d", i),
			Doc: "padding content to trigger segment rotation",
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	segCount := w.SegmentCount()
	if segCount < 2 {
		t.Fatalf("expected multiple segments, got %d", segCount)
	}
	t.Logf("created %d segments", segCount)

	w.Close()

	// Truncate the last segment to simulate crash during write
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	lastSeg := segFiles[len(segFiles)-1]
	info, _ := os.Stat(lastSeg)
	// Truncate to half
	os.Truncate(lastSeg, info.Size()/2)

	// Recover
	w2, err := Open(config)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Logf("ReadAll partial: %v", err)
	}

	// We should have recovered most entries from complete segments
	t.Logf("recovered %d / %d entries across %d segments", len(entries), total, segCount)
	if len(entries) == 0 {
		t.Fatal("recovered 0 entries")
	}

	// Entries should be in LSN order with no gaps up to some point
	for i := 0; i < len(entries); i++ {
		if entries[i].LSN != uint64(i+1) {
			t.Errorf("entry %d: LSN = %d, want %d", i, entries[i].LSN, i+1)
			break
		}
	}

	// Writing after recovery should work
	if err := w2.Write(&Entry{Op: OpInsert, ID: "post-crash"}); err != nil {
		t.Errorf("write after crash recovery failed: %v", err)
	}
}

// TestCrashRecovery_WriteAfterRecovery verifies that new writes after crash
// recovery get correct LSNs and coexist with recovered data.
func TestCrashRecovery_WriteAfterRecovery(t *testing.T) {
	dir := t.TempDir()

	// Phase 1: initial writes
	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for i := 0; i < 10; i++ {
		w.Write(&Entry{Op: OpInsert, Collection: "phase1", ID: fmt.Sprintf("d%d", i)})
	}
	w.Close()

	// Phase 2: recover and write more
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	for i := 0; i < 10; i++ {
		w2.Write(&Entry{Op: OpInsert, Collection: "phase2", ID: fmt.Sprintf("d%d", i)})
	}
	w2.Close()

	// Phase 3: recover again and write more
	w3, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen2: %v", err)
	}
	for i := 0; i < 5; i++ {
		w3.Write(&Entry{Op: OpUpdate, Collection: "phase3", ID: fmt.Sprintf("d%d", i)})
	}
	w3.Close()

	// Final check
	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(entries) != 25 {
		t.Fatalf("total entries = %d, want 25", len(entries))
	}

	// LSNs should be 1..25 with no gaps
	for i, e := range entries {
		if e.LSN != uint64(i+1) {
			t.Errorf("entry %d: LSN = %d, want %d", i, e.LSN, i+1)
		}
	}

	// Verify collection distribution
	counts := make(map[string]int)
	for _, e := range entries {
		counts[e.Collection]++
	}
	if counts["phase1"] != 10 || counts["phase2"] != 10 || counts["phase3"] != 5 {
		t.Errorf("collection distribution: %v", counts)
	}
}

// TestCrashRecovery_ZeroLengthEntry writes a valid header with length=0,
// which would produce an empty JSON body. Verify graceful handling.
func TestCrashRecovery_ZeroLengthEntry(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for i := 0; i < 3; i++ {
		w.Write(&Entry{Op: OpInsert, ID: fmt.Sprintf("doc-%d", i)})
	}
	w.Close()

	// Append a zero-length entry (header says 0 bytes of data)
	segFiles, _ := filepath.Glob(filepath.Join(dir, "wal-*.log"))
	lastSeg := segFiles[len(segFiles)-1]
	f, err := os.OpenFile(lastSeg, os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		t.Fatalf("open append: %v", err)
	}
	header := make([]byte, 8)
	binary.LittleEndian.PutUint32(header[0:4], 0) // length = 0
	emptyData := []byte{}
	binary.LittleEndian.PutUint32(header[4:8], crc32.ChecksumIEEE(emptyData))
	f.Write(header)
	f.Close()

	// Should not panic
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Logf("open after zero-length entry: %v (acceptable)", err)
		return
	}
	defer w2.Close()

	entries, _ := ReadAll(dir)
	// At least our 3 valid entries should be recovered
	if len(entries) < 3 {
		t.Errorf("recovered %d entries, want at least 3", len(entries))
	}
	t.Logf("recovered %d entries with zero-length entry appended", len(entries))
}

// TestCrashRecovery_ConcurrentWriteWithRotation tests crash recovery when
// concurrent writers are active and segment rotation occurs.
func TestCrashRecovery_ConcurrentWriteWithRotation(t *testing.T) {
	dir := t.TempDir()

	config := Config{
		Dir:            dir,
		MaxSegmentSize: 512, // trigger frequent rotation
		SyncOnWrite:    true,
	}

	w, err := Open(config)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	const writers = 4
	const writesPerWriter = 50
	const totalWrites = writers * writesPerWriter

	var wg sync.WaitGroup
	var writeErrors atomic.Int64

	for g := 0; g < writers; g++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()
			for j := 0; j < writesPerWriter; j++ {
				if err := w.Write(&Entry{
					Op:         OpInsert,
					Collection: fmt.Sprintf("writer-%d", wid),
					ID:         fmt.Sprintf("d%d", j),
					Doc:        "padding to trigger segment rotation under concurrent load",
				}); err != nil {
					writeErrors.Add(1)
				}
			}
		}(g)
	}

	wg.Wait()

	if n := writeErrors.Load(); n != 0 {
		t.Fatalf("%d write errors", n)
	}

	segCount := w.SegmentCount()
	t.Logf("concurrent writes created %d segments", segCount)
	w.Close()

	// Recover
	w2, err := Open(config)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	if w2.LSN() != totalWrites {
		t.Errorf("recovered LSN = %d, want %d", w2.LSN(), totalWrites)
	}

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(entries) != totalWrites {
		t.Errorf("recovered %d entries, want %d", len(entries), totalWrites)
	}
}

// TestCrashRecovery_VectorDataIntegrity verifies that float32 vector data
// survives the write-crash-recover cycle without corruption.
func TestCrashRecovery_VectorDataIntegrity(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	// Write entries with specific vector values
	dims := 128
	vectors := make([][]float32, 100)
	for i := range vectors {
		vec := make([]float32, dims)
		for d := 0; d < dims; d++ {
			vec[d] = float32(i*dims+d) * 0.001
		}
		vectors[i] = vec
		if err := w.Write(&Entry{
			Op:     OpInsert,
			ID:     fmt.Sprintf("vec-%d", i),
			Vector: vec,
		}); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	w.Close()

	// Recover
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	entries, err := ReadAll(dir)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(entries) != 100 {
		t.Fatalf("recovered %d entries, want 100", len(entries))
	}

	for i, e := range entries {
		if len(e.Vector) != dims {
			t.Fatalf("entry %d: vector dims = %d, want %d", i, len(e.Vector), dims)
		}
		for d := 0; d < dims; d++ {
			if e.Vector[d] != vectors[i][d] {
				t.Errorf("entry %d, dim %d: got %f, want %f", i, d, e.Vector[d], vectors[i][d])
				break
			}
		}
	}
}

// TestCrashRecovery_ReplayFromAfterCrash verifies that ReplayFrom works
// correctly after crash recovery, starting from a specific LSN.
func TestCrashRecovery_ReplayFromAfterCrash(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	for i := 0; i < 20; i++ {
		w.Write(&Entry{
			Op: OpInsert,
			ID: fmt.Sprintf("doc-%d", i),
		})
	}
	w.Close()

	// Reopen (simulating recovery)
	w2, err := Open(DefaultConfig(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()

	// Replay from LSN 11 — should yield entries 11..20
	var replayed []*Entry
	err = ReplayFrom(dir, 11, func(e *Entry) bool {
		replayed = append(replayed, e)
		return true
	})
	if err != nil {
		t.Fatalf("ReplayFrom: %v", err)
	}

	if len(replayed) != 10 {
		t.Fatalf("replayed %d entries from LSN 11, want 10", len(replayed))
	}

	for i, e := range replayed {
		wantLSN := uint64(11 + i)
		if e.LSN != wantLSN {
			t.Errorf("replayed[%d]: LSN = %d, want %d", i, e.LSN, wantLSN)
		}
	}
}

// readAllFromReader is a test helper that reads all entries via a Reader,
// returning partial results on error (used in corruption tests).
func readAllFromReader(r io.Reader) ([]*Entry, error) {
	reader := NewReader(r)
	var entries []*Entry
	for {
		entry, err := reader.Next()
		if err == io.EOF {
			return entries, nil
		}
		if err != nil {
			return entries, err
		}
		entries = append(entries, entry)
	}
}
