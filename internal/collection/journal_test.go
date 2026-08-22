package collection

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var collectionJournalTestStoreID = [16]byte{
	0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
	0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
}

func collectionJournalTestPaths(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	return filepath.Join(dir, "collections.journal"), filepath.Join(dir, "collections.journal.frozen")
}

func mustCollectionJournalFrame(t *testing.T, storeID [16]byte, lsn uint64, payload string) []byte {
	t.Helper()
	frame, err := encodeCollectionJournalFrame(storeID, lsn, []byte(payload), collectionJournalMaxPayload)
	if err != nil {
		t.Fatalf("encode journal frame: %v", err)
	}
	return frame
}

func writeCollectionJournalFramesForTest(t *testing.T, path string, frames ...[]byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		t.Fatalf("create journal fixture %q: %v", path, err)
	}
	for _, frame := range frames {
		if _, err := f.Write(frame); err != nil {
			_ = f.Close()
			t.Fatalf("write journal fixture %q: %v", path, err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close journal fixture %q: %v", path, err)
	}
}

func refreshCollectionJournalChecksum(frame []byte) {
	h := sha256.New()
	_, _ = h.Write(frame[:collectionJournalChecksumOffset])
	_, _ = h.Write(frame[int(collectionJournalHeaderSize):])
	copy(frame[collectionJournalChecksumOffset:int(collectionJournalHeaderSize)], h.Sum(nil))
}

func TestCollectionJournalAppendIsSyncedSecureAndReplayable(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, records, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open empty journal: %v", err)
	}
	if len(records) != 0 {
		t.Fatalf("empty journal returned %d records", len(records))
	}

	defaultSyncFile := j.ops.syncFile
	defaultSyncDir := j.ops.syncDir
	var fileSyncs, dirSyncs int
	j.ops.syncFile = func(f *os.File) error {
		fileSyncs++
		return defaultSyncFile(f)
	}
	j.ops.syncDir = func(path string) error {
		dirSyncs++
		return defaultSyncDir(path)
	}

	payload := []byte("create-collection")
	record, err := j.append(payload)
	if err != nil {
		t.Fatalf("append first frame: %v", err)
	}
	if record.LSN != 1 || string(record.Payload) != "create-collection" {
		t.Fatalf("first record = %#v", record)
	}
	payload[0] = 'X'
	if string(record.Payload) != "create-collection" {
		t.Fatal("append result aliases caller payload")
	}
	if fileSyncs != 1 || dirSyncs != 1 {
		t.Fatalf("first append syncs: file=%d dir=%d, want 1/1", fileSyncs, dirSyncs)
	}

	info, err := os.Stat(current)
	if err != nil {
		t.Fatalf("stat journal: %v", err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("journal permissions = %04o, want 0600", got)
	}

	if _, err := j.append([]byte("insert-document")); err != nil {
		t.Fatalf("append second frame: %v", err)
	}
	// The second append must still fsync file contents, but the directory
	// entry was made durable by this writer's first append; the once-per-
	// descriptor contract means no further namespace barrier is required.
	if fileSyncs != 2 || dirSyncs != 1 {
		t.Fatalf("second append syncs: file=%d dir=%d, want 2/1", fileSyncs, dirSyncs)
	}

	reopened, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("reopen journal: %v", err)
	}
	if len(replay) != 2 || replay[0].LSN != 1 || replay[1].LSN != 2 {
		t.Fatalf("replay = %#v", replay)
	}
	if string(replay[0].Payload) != "create-collection" || string(replay[1].Payload) != "insert-document" {
		t.Fatalf("replay payloads = %q, %q", replay[0].Payload, replay[1].Payload)
	}
	// A fresh descriptor in a new writer must re-establish namespace
	// durability before acknowledging its first frame: the previous process
	// may have crashed before its own directory sync survived.
	defaultReopenedSyncDir := reopened.ops.syncDir
	reopened.ops.syncDir = func(path string) error {
		dirSyncs++
		return defaultReopenedSyncDir(path)
	}
	third, err := reopened.append([]byte("delete-document"))
	if err != nil {
		t.Fatalf("append after reopen: %v", err)
	}
	if third.LSN != 3 {
		t.Fatalf("LSN after reopen = %d, want 3", third.LSN)
	}
	// A fresh descriptor in a new writer must re-establish namespace
	// durability before acknowledging its first frame: the previous process
	// may have crashed before its own directory sync survived.
	if dirSyncs != 2 {
		t.Fatalf("dir syncs after reopen+append = %d, want 2 (one per writer)", dirSyncs)
	}
}

func TestCollectionJournalParserRejectsMalformedFrames(t *testing.T) {
	valid := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "payload")
	otherStoreID := collectionJournalTestStoreID
	otherStoreID[0] ^= 0xff

	badChecksum := append([]byte(nil), valid...)
	badChecksum[len(badChecksum)-1] ^= 0xff
	unknownVersion := append([]byte(nil), valid...)
	binary.BigEndian.PutUint16(unknownVersion[collectionJournalVersionOffset:], collectionJournalVersion+1)
	wrongStore := mustCollectionJournalFrame(t, otherStoreID, 1, "payload")
	oversized := append([]byte(nil), valid...)
	binary.BigEndian.PutUint32(oversized[collectionJournalPayloadLenOffset:], collectionJournalMaxPayload+1)
	trailingJunk := append(append([]byte(nil), valid...), 0xde, 0xad)
	badMagic := append([]byte(nil), valid...)
	badMagic[0] ^= 0xff
	badHeaderSize := append([]byte(nil), valid...)
	binary.BigEndian.PutUint16(badHeaderSize[collectionJournalHeaderSizeOffset:], collectionJournalHeaderSize+1)
	zeroLSN := append([]byte(nil), valid...)
	binary.BigEndian.PutUint64(zeroLSN[collectionJournalLSNOffset:], 0)
	refreshCollectionJournalChecksum(zeroLSN)

	tests := []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{name: "checksum", data: badChecksum, wantErr: "checksum mismatch"},
		{name: "version", data: unknownVersion, wantErr: "unknown version"},
		{name: "store UUID", data: wrongStore, wantErr: "store UUID mismatch"},
		{name: "oversized", data: oversized, wantErr: "maximum"},
		{name: "trailing junk", data: trailingJunk, wantErr: "does not match the expected frame prefix"},
		{name: "magic", data: badMagic, wantErr: "invalid magic"},
		{name: "header size", data: badHeaderSize, wantErr: "unsupported header size"},
		{name: "zero LSN", data: zeroLSN, wantErr: "zero LSN"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			writeCollectionJournalFramesForTest(t, current, tc.data)
			j, records, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("open malformed journal: journal=%v records=%v err=%v, want %q", j, records, err, tc.wantErr)
			}
			if j != nil || records != nil {
				t.Fatalf("malformed journal leaked partial result: journal=%v records=%v", j, records)
			}
		})
	}
}

func TestCollectionJournalRepairsCurrentTerminalPartialTailExactlyOnce(t *testing.T) {
	first := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "first")
	second := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "second-payload")

	for cut := 1; cut < len(second); cut++ {
		t.Run(fmt.Sprintf("cut-%03d", cut), func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			writeCollectionJournalFramesForTest(t, current, first, second[:cut])

			journal, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err != nil {
				t.Fatalf("open with terminal partial tail: %v", err)
			}
			if len(replay) != 1 || replay[0].LSN != 1 || string(replay[0].Payload) != "first" {
				t.Fatalf("replay after repair = %#v", replay)
			}
			info, err := os.Stat(current)
			if err != nil {
				t.Fatal(err)
			}
			if info.Size() != int64(len(first)) {
				t.Fatalf("repaired journal size = %d, want %d", info.Size(), len(first))
			}

			appended, err := journal.append([]byte("second-payload"))
			if err != nil {
				t.Fatalf("append after repair: %v", err)
			}
			if appended.LSN != 2 {
				t.Fatalf("append LSN after repair = %d, want 2", appended.LSN)
			}

			_, reopened, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err != nil {
				t.Fatalf("second reopen: %v", err)
			}
			if len(reopened) != 2 || reopened[0].LSN != 1 || reopened[1].LSN != 2 {
				t.Fatalf("replay after append/reopen = %#v", reopened)
			}
			if string(reopened[0].Payload) != "first" || string(reopened[1].Payload) != "second-payload" {
				t.Fatalf("replayed payloads = %q, %q", reopened[0].Payload, reopened[1].Payload)
			}
		})
	}
}

func TestCollectionJournalRepairsPartialFirstFrameToEmpty(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	first := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "first")
	writeCollectionJournalFramesForTest(t, current, first[:17])

	journal, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open partial first frame: %v", err)
	}
	if len(replay) != 0 {
		t.Fatalf("partial first frame replayed records: %#v", replay)
	}
	info, err := os.Stat(current)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 0 {
		t.Fatalf("repaired first-frame journal size = %d, want 0", info.Size())
	}
	record, err := journal.append([]byte("first"))
	if err != nil {
		t.Fatal(err)
	}
	if record.LSN != 1 {
		t.Fatalf("first append after repair LSN = %d, want 1", record.LSN)
	}
}

func TestCollectionJournalPartialTailRepairRemainsFailClosed(t *testing.T) {
	first := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "first")
	second := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "second")
	third := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 3, "third")
	wrongPrefix := append([]byte(nil), second[:17]...)
	wrongPrefix[0] ^= 0xff

	tests := []struct {
		name    string
		current []byte
		frozen  []byte
		wantErr string
	}{
		{
			name:    "frozen partial header",
			frozen:  first[:17],
			wantErr: "partial frame header",
		},
		{
			name:    "frozen partial body",
			frozen:  first[:len(first)-1],
			wantErr: "partial frame body",
		},
		{
			name:    "current non-prefix junk",
			current: append(append([]byte(nil), first...), wrongPrefix...),
			wantErr: "does not match the expected frame prefix",
		},
		{
			name:    "current partial LSN gap",
			current: append(append([]byte(nil), first...), third[:len(third)-1]...),
			wantErr: "expected frame prefix for LSN 2",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			if tc.current != nil {
				writeCollectionJournalFramesForTest(t, current, tc.current)
			}
			if tc.frozen != nil {
				writeCollectionJournalFramesForTest(t, frozen, tc.frozen)
			}
			journal, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("unsafe partial tail open: journal=%v replay=%v err=%v, want %q", journal, replay, err, tc.wantErr)
			}
			if journal != nil || replay != nil {
				t.Fatalf("unsafe partial tail returned usable state: journal=%v replay=%v", journal, replay)
			}
		})
	}
}

func TestCollectionJournalPartialTailRepairFailuresFailClosed(t *testing.T) {
	tests := []struct {
		name   string
		inject func(*collectionJournal)
	}{
		{
			name: "truncate",
			inject: func(j *collectionJournal) {
				j.ops.truncateFile = func(*os.File, int64) error { return errors.New("injected truncate failure") }
			},
		},
		{
			name: "file sync",
			inject: func(j *collectionJournal) {
				j.ops.syncFile = func(*os.File) error { return errors.New("injected file sync failure") }
			},
		},
		{
			name: "parent sync",
			inject: func(j *collectionJournal) {
				j.ops.syncDir = func(string) error { return errors.New("injected parent sync failure") }
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			first := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "first")
			second := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "second")
			writeCollectionJournalFramesForTest(t, current, first, second[:len(second)-1])
			journal, err := newCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err != nil {
				t.Fatal(err)
			}
			tc.inject(journal)
			if replay, err := journal.readAfter(0); err == nil || !strings.Contains(err.Error(), "injected") {
				t.Fatalf("repair failure did not fail closed: replay=%v err=%v", replay, err)
			}
		})
	}
}

func TestCollectionJournalRejectsInsecureArtifactPermissions(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, current, mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "payload"))
	if err := os.Chmod(current, 0o644); err != nil {
		t.Fatalf("broaden fixture permissions: %v", err)
	}
	if _, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0); err == nil || !strings.Contains(err.Error(), "permissions 0644") {
		t.Fatalf("insecure journal permissions must fail closed: %v", err)
	}
}

func TestCollectionJournalRejectsSequenceViolations(t *testing.T) {
	tests := []struct {
		name        string
		frozenLSNs  []uint64
		currentLSNs []uint64
		appliedLSN  uint64
		wantErr     string
	}{
		{name: "duplicate in one file", currentLSNs: []uint64{1, 1}, wantErr: "duplicates or regresses"},
		{name: "duplicate across files", frozenLSNs: []uint64{1}, currentLSNs: []uint64{1}, wantErr: "duplicates or regresses"},
		{name: "regression across files", frozenLSNs: []uint64{2}, currentLSNs: []uint64{1}, appliedLSN: 1, wantErr: "duplicates or regresses"},
		{name: "gap in one file", currentLSNs: []uint64{1, 3}, wantErr: "LSN gap"},
		{name: "gap across files", frozenLSNs: []uint64{1}, currentLSNs: []uint64{3}, wantErr: "LSN gap"},
		{name: "gap after checkpoint", currentLSNs: []uint64{3}, appliedLSN: 1, wantErr: "gap after checkpoint"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			if len(tc.frozenLSNs) > 0 {
				frames := make([][]byte, 0, len(tc.frozenLSNs))
				for _, lsn := range tc.frozenLSNs {
					frames = append(frames, mustCollectionJournalFrame(t, collectionJournalTestStoreID, lsn, fmt.Sprintf("frozen-%d", lsn)))
				}
				writeCollectionJournalFramesForTest(t, frozen, frames...)
			}
			if len(tc.currentLSNs) > 0 {
				frames := make([][]byte, 0, len(tc.currentLSNs))
				for _, lsn := range tc.currentLSNs {
					frames = append(frames, mustCollectionJournalFrame(t, collectionJournalTestStoreID, lsn, fmt.Sprintf("current-%d", lsn)))
				}
				writeCollectionJournalFramesForTest(t, current, frames...)
			}

			if _, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, tc.appliedLSN); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("sequence violation did not fail with %q: %v", tc.wantErr, err)
			}
		})
	}
}

func TestCollectionJournalReadsFrozenThenCurrentAndSkipsCheckpointedFrames(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, frozen,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "stale-one"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "stale-two"),
	)
	writeCollectionJournalFramesForTest(t, current,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 3, "replay-three"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 4, "replay-four"),
	)

	j, records, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 2)
	if err != nil {
		t.Fatalf("read frozen/current journal: %v", err)
	}
	if len(records) != 2 || records[0].LSN != 3 || records[1].LSN != 4 {
		t.Fatalf("replay records = %#v", records)
	}
	if string(records[0].Payload) != "replay-three" || string(records[1].Payload) != "replay-four" {
		t.Fatalf("replay order = %q then %q", records[0].Payload, records[1].Payload)
	}

	next, err := j.append([]byte("new-current-frame"))
	if err != nil {
		t.Fatalf("append after ordered recovery: %v", err)
	}
	if next.LSN != 5 {
		t.Fatalf("next recovered LSN = %d, want 5", next.LSN)
	}
}

func TestCollectionJournalAllowsCheckpointCoveredCleanupGap(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, frozen,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "stale-1"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "stale-2"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 3, "stale-3"),
	)
	// Snapshot LSN 5 covers the missing 4..5 records after a partial cleanup
	// removed current but failed to remove frozen.
	j, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 5)
	if err != nil {
		t.Fatalf("open partially cleaned journal: %v", err)
	}
	if len(replay) != 0 {
		t.Fatalf("stale frozen journal produced replay: %#v", replay)
	}
	record, err := j.append([]byte("acknowledged-after-restart"))
	if err != nil {
		t.Fatalf("append after partial cleanup restart: %v", err)
	}
	if record.LSN != 6 {
		t.Fatalf("new LSN = %d, want 6", record.LSN)
	}

	_, replay, err = openCollectionJournal(current, frozen, collectionJournalTestStoreID, 5)
	if err != nil {
		t.Fatalf("reopen after acknowledged append: %v", err)
	}
	if len(replay) != 1 || replay[0].LSN != 6 || string(replay[0].Payload) != "acknowledged-after-restart" {
		t.Fatalf("replay after covered gap = %#v", replay)
	}
}

func TestCollectionJournalRemovesCheckpointCoveredCurrentBeforeAppend(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, current,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "stale-1"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "stale-2"),
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 3, "stale-3"),
	)

	j, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 5)
	if err != nil {
		t.Fatalf("open stale current journal: %v", err)
	}
	if len(replay) != 0 {
		t.Fatalf("stale current journal produced replay: %#v", replay)
	}
	if _, err := os.Stat(current); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("open retained stale current journal: %v", err)
	}
	record, err := j.append([]byte("acknowledged-after-stale-current"))
	if err != nil {
		t.Fatalf("append after stale-current cleanup: %v", err)
	}
	if record.LSN != 6 {
		t.Fatalf("new LSN = %d, want 6", record.LSN)
	}
	_, replay, err = openCollectionJournal(current, frozen, collectionJournalTestStoreID, 5)
	if err != nil {
		t.Fatalf("reopen after stale-current append: %v", err)
	}
	if len(replay) != 1 || replay[0].LSN != 6 {
		t.Fatalf("replay after stale-current cleanup = %#v", replay)
	}
}

func TestCollectionJournalValidatesAllArtifactsBeforeReplay(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, frozen, mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "must-not-leak"))
	corrupt := mustCollectionJournalFrame(t, collectionJournalTestStoreID, 2, "corrupt")
	corrupt[len(corrupt)-1] ^= 0xff
	writeCollectionJournalFramesForTest(t, current, corrupt)

	j, err := newCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("construct journal: %v", err)
	}
	records, err := j.readAfter(0)
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("corrupt current artifact must fail: records=%#v err=%v", records, err)
	}
	if records != nil {
		t.Fatalf("corrupt current artifact leaked frozen replay: %#v", records)
	}
}

func TestCollectionJournalPayloadBoundIsEnforcedBeforeIO(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	j.maxPayload = 8
	if _, err := j.append(make([]byte, 9)); err == nil || !strings.Contains(err.Error(), "maximum is 8") {
		t.Fatalf("oversized append error = %v", err)
	}
	if _, err := os.Stat(current); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("oversized append touched journal: %v", err)
	}
	if fault := j.writeFault(); fault != nil {
		t.Fatalf("pre-I/O validation faulted writer: %v", fault)
	}
	if _, err := j.append(make([]byte, 8)); err != nil {
		t.Fatalf("bounded append after rejection: %v", err)
	}
}

func TestCollectionJournalRotateIsDurableAndNeverOverwritesFrozen(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("first")); err != nil {
		t.Fatalf("append first frame: %v", err)
	}

	defaultSyncDir := j.ops.syncDir
	dirSyncs := 0
	j.ops.syncDir = func(path string) error {
		dirSyncs++
		return defaultSyncDir(path)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate journal: %v", err)
	}
	if dirSyncs != 2 {
		t.Fatalf("rotation directory syncs = %d, want 2", dirSyncs)
	}
	if _, err := os.Stat(current); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("current artifact remains after rotation: %v", err)
	}
	if info, err := os.Stat(frozen); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("frozen artifact stat: info=%v err=%v", info, err)
	}

	j2, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open rotated journal: %v", err)
	}
	if len(replay) != 1 || replay[0].LSN != 1 {
		t.Fatalf("rotated replay = %#v", replay)
	}
	if _, err := j2.append([]byte("second")); err != nil {
		t.Fatalf("append new current frame: %v", err)
	}
	frozenBefore, err := os.ReadFile(frozen)
	if err != nil {
		t.Fatalf("read frozen before refused rotation: %v", err)
	}
	currentBefore, err := os.ReadFile(current)
	if err != nil {
		t.Fatalf("read current before refused rotation: %v", err)
	}

	if err := j2.rotate(); !errors.Is(err, errCollectionJournalFrozenExists) {
		t.Fatalf("second rotation error = %v, want frozen-exists", err)
	}
	if got, _ := os.ReadFile(frozen); string(got) != string(frozenBefore) {
		t.Fatal("refused rotation overwrote frozen artifact")
	}
	if got, _ := os.ReadFile(current); string(got) != string(currentBefore) {
		t.Fatal("refused rotation changed current artifact")
	}
}

func TestCollectionJournalRepairsInterruptedPortableRotation(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, current,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "interrupted"),
	)
	if err := os.Link(current, frozen); err != nil {
		t.Fatalf("create interrupted rotation fixture: %v", err)
	}

	j, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("repair interrupted rotation: %v", err)
	}
	if len(replay) != 1 || replay[0].LSN != 1 || string(replay[0].Payload) != "interrupted" {
		t.Fatalf("replay after rotation repair = %#v", replay)
	}
	if _, err := os.Stat(current); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("rotation repair retained current hard link: %v", err)
	}
	if _, err := os.Stat(frozen); err != nil {
		t.Fatalf("rotation repair lost frozen artifact: %v", err)
	}
	next, err := j.append([]byte("after-repair"))
	if err != nil {
		t.Fatalf("append after rotation repair: %v", err)
	}
	if next.LSN != 2 {
		t.Fatalf("next LSN after rotation repair = %d, want 2", next.LSN)
	}
}

func TestCollectionJournalRepairSyncsFrozenLinkBeforeUnlink(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	writeCollectionJournalFramesForTest(t, current,
		mustCollectionJournalFrame(t, collectionJournalTestStoreID, 1, "interrupted"),
	)
	if err := os.Link(current, frozen); err != nil {
		t.Fatal(err)
	}
	j, err := newCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatal(err)
	}
	j.ops.syncDir = func(string) error { return errors.New("injected pre-unlink sync failure") }
	if _, err := j.readAfter(0); err == nil || !strings.Contains(err.Error(), "pre-unlink sync failure") {
		t.Fatalf("repair pre-unlink sync error = %v", err)
	}
	if _, err := os.Stat(current); err != nil {
		t.Fatalf("repair unlinked current before frozen link was durable: %v", err)
	}
	if _, err := os.Stat(frozen); err != nil {
		t.Fatalf("repair lost frozen link: %v", err)
	}
}

func TestCollectionJournalRotationFaultsAreLatched(t *testing.T) {
	tests := []struct {
		name   string
		inject func(*collectionJournal)
	}{
		{
			name: "link uncertainty",
			inject: func(j *collectionJournal) {
				link := j.ops.linkNoReplace
				j.ops.linkNoReplace = func(oldPath, newPath string) error {
					if err := link(oldPath, newPath); err != nil {
						return err
					}
					return errors.New("injected link uncertainty")
				}
			},
		},
		{
			name: "frozen link directory sync uncertainty",
			inject: func(j *collectionJournal) {
				j.ops.syncDir = func(string) error { return errors.New("injected rotation sync uncertainty") }
			},
		},
		{
			name: "current name removal uncertainty",
			inject: func(j *collectionJournal) {
				j.ops.removePath = func(string) error { return errors.New("injected rotation remove uncertainty") }
			},
		},
		{
			name: "current removal directory sync uncertainty",
			inject: func(j *collectionJournal) {
				defaultSync := j.ops.syncDir
				calls := 0
				j.ops.syncDir = func(path string) error {
					calls++
					if calls == 2 {
						return errors.New("injected second rotation sync uncertainty")
					}
					return defaultSync(path)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err != nil {
				t.Fatalf("open journal: %v", err)
			}
			if _, err := j.append([]byte("frame")); err != nil {
				t.Fatalf("append frame: %v", err)
			}
			tc.inject(j)
			if err := j.rotate(); err == nil || !strings.Contains(err.Error(), "indeterminate") {
				t.Fatalf("injected rotation error = %v", err)
			}
			if fault := j.writeFault(); fault == nil {
				t.Fatal("rotation uncertainty did not latch fault")
			}
			if _, err := j.append([]byte("must-not-write")); !errors.Is(err, errCollectionJournalFaulted) {
				t.Fatalf("faulted rotation allowed append: %v", err)
			}
		})
	}
}

func TestCollectionJournalCleanupIsDurableAndRetryable(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("checkpointed")); err != nil {
		t.Fatalf("append frame: %v", err)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate frame: %v", err)
	}

	j.ops.syncDir = func(string) error { return errors.New("injected cleanup sync failure") }
	if err := j.cleanupFrozen(); err == nil || !strings.Contains(err.Error(), "cleanup") {
		t.Fatalf("cleanup sync failure = %v", err)
	}
	if _, err := os.Stat(frozen); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cleanup did not unlink before injected sync failure: %v", err)
	}

	retrySyncs := 0
	defaultSyncDir := defaultCollectionJournalFileOps().syncDir
	j.ops.syncDir = func(path string) error {
		retrySyncs++
		return defaultSyncDir(path)
	}
	if err := j.cleanupFrozen(); err != nil {
		t.Fatalf("retry cleanup after unlink: %v", err)
	}
	if retrySyncs != 1 {
		t.Fatalf("cleanup retry directory syncs = %d, want 1", retrySyncs)
	}
}

func TestCollectionJournalCleanupPreservesArtifactOnRemoveFailure(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("checkpointed")); err != nil {
		t.Fatalf("append frame: %v", err)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate frame: %v", err)
	}

	dirSyncs := 0
	j.ops.removePath = func(string) error { return errors.New("injected remove failure") }
	j.ops.syncDir = func(string) error {
		dirSyncs++
		return nil
	}
	if err := j.cleanupFrozen(); err == nil || !strings.Contains(err.Error(), "remove frozen") {
		t.Fatalf("remove failure = %v", err)
	}
	if dirSyncs != 0 {
		t.Fatalf("directory synced despite failed unlink: %d", dirSyncs)
	}
	if _, err := os.Stat(frozen); err != nil {
		t.Fatalf("remove failure did not preserve artifact: %v", err)
	}
}

func TestCollectionJournalCleanupAllRemovesBothAndSyncsAbsentRetries(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("frozen-frame")); err != nil {
		t.Fatalf("append frozen frame: %v", err)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate frozen frame: %v", err)
	}
	if _, err := j.append([]byte("current-frame")); err != nil {
		t.Fatalf("append current frame: %v", err)
	}

	defaultSyncDir := j.ops.syncDir
	dirSyncs := 0
	j.ops.syncDir = func(path string) error {
		dirSyncs++
		return defaultSyncDir(path)
	}
	if err := j.cleanupAll(); err != nil {
		t.Fatalf("cleanup all artifacts: %v", err)
	}
	for _, path := range []string{frozen, current} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("cleanup all left %q: %v", path, err)
		}
	}
	if dirSyncs != 1 {
		t.Fatalf("cleanup all directory syncs = %d, want 1", dirSyncs)
	}

	// A retry after names are already absent must still make a prior unlink
	// durable rather than treating os.ErrNotExist as proof of durability.
	if err := j.cleanupAll(); err != nil {
		t.Fatalf("retry cleanup with absent artifacts: %v", err)
	}
	if dirSyncs != 2 {
		t.Fatalf("absent cleanup retry directory syncs = %d, want 2", dirSyncs)
	}
}

func TestCollectionJournalCleanupAllSyncFailureIsRetryable(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("frozen-frame")); err != nil {
		t.Fatalf("append frozen frame: %v", err)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate frozen frame: %v", err)
	}
	if _, err := j.append([]byte("current-frame")); err != nil {
		t.Fatalf("append current frame: %v", err)
	}

	j.ops.syncDir = func(string) error { return errors.New("injected cleanup-all sync failure") }
	if err := j.cleanupAll(); err == nil || !strings.Contains(err.Error(), "cleanup-all sync failure") {
		t.Fatalf("cleanup-all sync failure = %v", err)
	}
	for _, path := range []string{frozen, current} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("cleanup-all sync failure did not unlink %q: %v", path, err)
		}
	}

	retrySyncs := 0
	defaultSyncDir := defaultCollectionJournalFileOps().syncDir
	j.ops.syncDir = func(path string) error {
		retrySyncs++
		return defaultSyncDir(path)
	}
	if err := j.cleanupAll(); err != nil {
		t.Fatalf("retry cleanup all after sync failure: %v", err)
	}
	if retrySyncs != 1 {
		t.Fatalf("cleanup-all retry directory syncs = %d, want 1", retrySyncs)
	}
}

func TestCollectionJournalCleanupAllContinuesAfterOneRemoveFailure(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := j.append([]byte("frozen-frame")); err != nil {
		t.Fatalf("append frozen frame: %v", err)
	}
	if err := j.rotate(); err != nil {
		t.Fatalf("rotate frozen frame: %v", err)
	}
	if _, err := j.append([]byte("current-frame")); err != nil {
		t.Fatalf("append current frame: %v", err)
	}

	defaultRemove := j.ops.removePath
	defaultSyncDir := j.ops.syncDir
	dirSyncs := 0
	j.ops.removePath = func(path string) error {
		if path == frozen {
			return errors.New("injected frozen remove failure")
		}
		return defaultRemove(path)
	}
	j.ops.syncDir = func(path string) error {
		dirSyncs++
		return defaultSyncDir(path)
	}
	if err := j.cleanupAll(); err == nil || !strings.Contains(err.Error(), "frozen remove failure") {
		t.Fatalf("partial cleanup-all error = %v", err)
	}
	if dirSyncs != 1 {
		t.Fatalf("partial cleanup-all directory syncs = %d, want 1", dirSyncs)
	}
	if _, err := os.Stat(frozen); err != nil {
		t.Fatalf("failed frozen removal did not preserve artifact: %v", err)
	}
	if _, err := os.Stat(current); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("successful current removal was not attempted: %v", err)
	}

	j.ops.removePath = defaultRemove
	if err := j.cleanupAll(); err != nil {
		t.Fatalf("retry partial cleanup all: %v", err)
	}
	if _, err := os.Stat(frozen); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("retry left frozen artifact: %v", err)
	}
}

func TestCollectionJournalAppendFaultsAreLatched(t *testing.T) {
	tests := []struct {
		name      string
		inject    func(*collectionJournal)
		viaRotate bool
	}{
		{
			name: "partial write",
			inject: func(j *collectionJournal) {
				j.ops.writeFile = func(f *os.File, p []byte) (int, error) {
					n, err := f.Write(p[:len(p)/2])
					if err != nil {
						return n, err
					}
					return n, errors.New("injected partial write")
				}
			},
		},
		{
			name: "file sync",
			inject: func(j *collectionJournal) {
				j.ops.syncFile = func(*os.File) error { return errors.New("injected file sync failure") }
			},
		},
		{
			name: "close during rotation",
			inject: func(j *collectionJournal) {
				j.ops.closeFile = func(f *os.File) error {
					if err := f.Close(); err != nil {
						return err
					}
					return errors.New("injected close uncertainty")
				}
			},
			// Close no longer runs inside append; descriptor teardown happens
			// when the artifact identity changes, so drive it via rotate.
			viaRotate: true,
		},
		{
			name: "creation directory sync",
			inject: func(j *collectionJournal) {
				j.ops.syncDir = func(string) error { return errors.New("injected creation sync failure") }
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			current, frozen := collectionJournalTestPaths(t)
			j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
			if err != nil {
				t.Fatalf("open journal: %v", err)
			}
			tc.inject(j)
			if tc.viaRotate {
				// Establish a cached descriptor first; close only runs during
				// identity changes when one exists.
				if _, err := j.append([]byte("seed-frame")); err != nil {
					t.Fatalf("seed append before rotation: %v", err)
				}
				if err := j.rotate(); err == nil || !strings.Contains(err.Error(), "indeterminate") {
					t.Fatalf("injected rotation error = %v", err)
				}
			} else {
				if _, err := j.append([]byte("uncertain-frame")); err == nil || !strings.Contains(err.Error(), "indeterminate") {
					t.Fatalf("injected append error = %v", err)
				}
			}
			if fault := j.writeFault(); fault == nil {
				t.Fatal("append uncertainty did not latch fault")
			}
			if _, err := j.append([]byte("must-not-write")); !errors.Is(err, errCollectionJournalFaulted) {
				t.Fatalf("faulted journal allowed another append: %v", err)
			}
		})
	}
}

func TestCollectionJournalPreIOFailureDoesNotPoisonWriter(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defaultOpen := j.ops.openFile
	j.ops.openFile = func(string, int, os.FileMode) (*os.File, error) {
		return nil, errors.New("injected open failure")
	}
	if _, err := j.append([]byte("not-written")); err == nil {
		t.Fatal("expected open failure")
	}
	if fault := j.writeFault(); fault != nil {
		t.Fatalf("pre-I/O open failure poisoned writer: %v", fault)
	}
	j.ops.openFile = defaultOpen
	if _, err := j.append([]byte("written-after-retry")); err != nil {
		t.Fatalf("retry append after open failure: %v", err)
	}
}

func TestCollectionJournalRetryAfterPostCreateSetupFailureSyncsNamespace(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}

	defaultChmod := j.ops.chmodFile
	chmodCalls := 0
	j.ops.chmodFile = func(f *os.File, mode os.FileMode) error {
		chmodCalls++
		if chmodCalls == 1 {
			return errors.New("injected post-create chmod failure")
		}
		return defaultChmod(f, mode)
	}

	defaultSyncDir := j.ops.syncDir
	dirSyncCalls := 0
	j.ops.syncDir = func(path string) error {
		dirSyncCalls++
		return defaultSyncDir(path)
	}

	if _, err := j.append([]byte("not-written")); err == nil {
		t.Fatal("expected post-create setup failure")
	}
	if fault := j.writeFault(); fault != nil {
		t.Fatalf("pre-I/O setup failure poisoned writer: %v", fault)
	}
	if _, err := j.append([]byte("written-after-retry")); err != nil {
		t.Fatalf("retry append after post-create setup failure: %v", err)
	}
	if dirSyncCalls != 1 {
		t.Fatalf("successful retry directory sync calls = %d; want 1", dirSyncCalls)
	}

	reopened, records, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("reopen journal: %v", err)
	}
	if reopened.lastLSN != 1 || len(records) != 1 || string(records[0].Payload) != "written-after-retry" {
		t.Fatalf("replayed records after retry = %+v, last LSN = %d", records, reopened.lastLSN)
	}
}

func TestCollectionJournalConcurrentAppendsRemainContiguous(t *testing.T) {
	current, frozen := collectionJournalTestPaths(t)
	j, _, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}

	const writers = 24
	results := make(chan collectionJournalRecord, writers)
	errs := make(chan error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			record, err := j.append([]byte(fmt.Sprintf("payload-%02d", i)))
			if err != nil {
				errs <- err
				return
			}
			results <- record
		}(i)
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		t.Errorf("concurrent append: %v", err)
	}
	if t.Failed() {
		return
	}

	seen := make(map[uint64]bool, writers)
	for record := range results {
		seen[record.LSN] = true
	}
	for lsn := uint64(1); lsn <= writers; lsn++ {
		if !seen[lsn] {
			t.Fatalf("concurrent append missing LSN %d: %v", lsn, seen)
		}
	}

	_, replay, err := openCollectionJournal(current, frozen, collectionJournalTestStoreID, 0)
	if err != nil {
		t.Fatalf("reopen concurrent journal: %v", err)
	}
	if len(replay) != writers {
		t.Fatalf("replayed %d concurrent records, want %d", len(replay), writers)
	}
	for i, record := range replay {
		if record.LSN != uint64(i+1) {
			t.Fatalf("replay record %d has LSN %d", i, record.LSN)
		}
	}
}

func TestCollectionJournalPathAndStoreInvariants(t *testing.T) {
	dir := t.TempDir()
	current := filepath.Join(dir, "journal")
	tests := []struct {
		name    string
		current string
		frozen  string
		storeID [16]byte
	}{
		{name: "same path", current: current, frozen: current, storeID: collectionJournalTestStoreID},
		{name: "different directories", current: current, frozen: filepath.Join(t.TempDir(), "frozen"), storeID: collectionJournalTestStoreID},
		{name: "zero store UUID", current: current, frozen: current + ".frozen"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := openCollectionJournal(tc.current, tc.frozen, tc.storeID, 0); err == nil {
				t.Fatal("invalid journal configuration succeeded")
			}
		})
	}
}
