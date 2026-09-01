package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/storage"
)

func writeModernWALRecords(t *testing.T, path string, entries ...walEntry) {
	t.Helper()

	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open WAL %q: %v", path, err)
	}
	enc := json.NewEncoder(f)
	for i := range entries {
		if entries[i].Version == 0 {
			entries[i].Version = currentWALVersion
		}
		if entries[i].Seq == 0 {
			_ = f.Close()
			t.Fatalf("modern WAL entry %d has no sequence", i)
		}
		checksum, err := walEntryChecksum(entries[i])
		if err != nil {
			_ = f.Close()
			t.Fatalf("checksum WAL entry %d: %v", i, err)
		}
		entries[i].Checksum = checksum
		if err := enc.Encode(&entries[i]); err != nil {
			_ = f.Close()
			t.Fatalf("encode WAL entry %d: %v", i, err)
		}
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		t.Fatalf("sync WAL %q: %v", path, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close WAL %q: %v", path, err)
	}
}

func TestWALOnlyStartupRecoversAndCheckpoints(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	id, err := writer.Add(
		[]float32{1, 2, 3},
		"durable document",
		"",
		map[string]string{"kind": "recovery"},
		"articles",
		"tenant-a",
	)
	if err != nil {
		t.Fatalf("append WAL-only mutation: %v", err)
	}
	if id != "doc-0" {
		t.Fatalf("first auto ID = %q, want doc-0", id)
	}
	if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
		t.Fatalf("snapshot unexpectedly exists before recovery: %v", err)
	}

	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil {
		t.Fatalf("recover WAL without snapshot: %v", err)
	}
	if !loaded {
		t.Fatal("WAL-only recovery must report existing state")
	}
	if recovered.Count != 1 || recovered.GetID(0) != "doc-0" || recovered.GetDoc(0) != "durable document" {
		t.Fatalf("unexpected recovered document state: count=%d id=%q doc=%q", recovered.Count, recovered.GetID(0), recovered.GetDoc(0))
	}
	hid := hashID("doc-0")
	if recovered.Coll[hid] != "articles" || recovered.TenantID[hid] != "tenant-a" || recovered.Meta[hid]["kind"] != "recovery" {
		t.Fatalf("recovered ownership/metadata mismatch: coll=%q tenant=%q meta=%v", recovered.Coll[hid], recovered.TenantID[hid], recovered.Meta[hid])
	}
	if _, err := os.Stat(snapshotPath); err != nil {
		t.Fatalf("recovery checkpoint missing: %v", err)
	}
	if _, err := os.Stat(walPath); !os.IsNotExist(err) {
		t.Fatalf("checkpointed WAL was not removed: %v", err)
	}

	restarted, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("load recovery checkpoint: loaded=%v err=%v", loaded, err)
	}
	if restarted.Count != 1 || restarted.GetDoc(0) != "durable document" {
		t.Fatalf("checkpoint did not retain recovered state: count=%d doc=%q", restarted.Count, restarted.GetDoc(0))
	}
	secondID, err := restarted.Add([]float32{3, 2, 1}, "second", "", nil, "articles", "tenant-a")
	if err != nil {
		t.Fatalf("add after WAL-only recovery: %v", err)
	}
	if secondID != "doc-1" {
		t.Fatalf("auto-ID high-water was not recovered: got %q, want doc-1", secondID)
	}
}

func TestWALCrashRecoverySubprocess(t *testing.T) {
	if os.Getenv("DEEPDATA_WAL_CRASH_CHILD") == "1" {
		snapshotPath := os.Getenv("DEEPDATA_WAL_CRASH_PATH")
		mode := os.Getenv("DEEPDATA_WAL_CRASH_MODE")
		store := NewVectorStore(10, 3)
		store.walPath = snapshotPath + ".wal"
		if _, err := store.Add([]float32{1, 2, 3}, "acked-before-kill", "id-1", map[string]string{"mode": mode}, "default", "tenant-a"); err != nil {
			_, _ = os.Stderr.WriteString("child add: " + err.Error() + "\n")
			os.Exit(2)
		}
		if mode == "snapshot-plus-wal" {
			if err := store.Save(snapshotPath); err != nil {
				_, _ = os.Stderr.WriteString("child save: " + err.Error() + "\n")
				os.Exit(3)
			}
		}
		_, _ = os.Stdout.WriteString("ACK\n")
		for {
			time.Sleep(time.Hour)
		}
	}

	for _, mode := range []string{"wal-only", "snapshot-plus-wal"} {
		t.Run(mode, func(t *testing.T) {
			snapshotPath := filepath.Join(t.TempDir(), "index.gob")
			cmd := exec.Command(os.Args[0], "-test.run=^TestWALCrashRecoverySubprocess$")
			cmd.Env = append(os.Environ(),
				"DEEPDATA_WAL_CRASH_CHILD=1",
				"DEEPDATA_WAL_CRASH_PATH="+snapshotPath,
				"DEEPDATA_WAL_CRASH_MODE="+mode,
			)
			stdout, err := cmd.StdoutPipe()
			if err != nil {
				t.Fatalf("child stdout: %v", err)
			}
			var stderr bytes.Buffer
			cmd.Stderr = &stderr
			if err := cmd.Start(); err != nil {
				t.Fatalf("start crash child: %v", err)
			}
			ack := make(chan string, 1)
			go func() {
				line, _ := bufio.NewReader(stdout).ReadString('\n')
				ack <- strings.TrimSpace(line)
			}()
			select {
			case line := <-ack:
				if line != "ACK" {
					_ = cmd.Process.Kill()
					_ = cmd.Wait()
					t.Fatalf("crash child did not acknowledge mutation: line=%q stderr=%s", line, stderr.String())
				}
			case <-time.After(10 * time.Second):
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				t.Fatalf("timed out waiting for crash child: %s", stderr.String())
			}
			if err := cmd.Process.Kill(); err != nil {
				t.Fatalf("SIGKILL crash child: %v", err)
			}
			if err := cmd.Wait(); err == nil {
				t.Fatal("crash child exited cleanly instead of being killed")
			}

			if _, err := os.Stat(snapshotPath + ".wal"); err != nil {
				t.Fatalf("acked mutation has no WAL after crash: %v", err)
			}
			recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
			if err != nil || !loaded {
				t.Fatalf("restart after SIGKILL: loaded=%v err=%v", loaded, err)
			}
			if recovered.Count != 1 || recovered.GetDoc(0) != "acked-before-kill" || recovered.TenantID[hashID("id-1")] != "tenant-a" {
				t.Fatalf("acked mutation did not survive SIGKILL: count=%d docs=%v tenants=%v", recovered.Count, recovered.Docs, recovered.TenantID)
			}
			if _, err := os.Stat(snapshotPath + ".wal"); !os.IsNotExist(err) {
				t.Fatalf("restart did not checkpoint recovered WAL: %v", err)
			}
		})
	}
}

func TestWALOnlyStartupRejectsInvalidArtifact(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	if err := os.WriteFile(walPath, []byte("{not-json\n"), 0o600); err != nil {
		t.Fatalf("write invalid WAL: %v", err)
	}

	store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err == nil || store != nil || loaded {
		t.Fatalf("invalid WAL must fail closed: store=%v loaded=%v err=%v", store, loaded, err)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("invalid WAL was not preserved: %v", err)
	}
	if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
		t.Fatalf("invalid WAL created a snapshot: %v", err)
	}
}

func TestLegacyWALOnlyRecoveryDerivesAutoIDHighWater(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	f, err := os.OpenFile(walPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("create legacy WAL: %v", err)
	}
	enc := json.NewEncoder(f)
	for _, entry := range []walEntry{
		{Op: "insert", ID: "doc-0", Doc: "first", Vec: []float32{1, 0, 0}},
		{Op: "insert", ID: "doc-1", Doc: "second", Vec: []float32{0, 1, 0}},
	} {
		if err := enc.Encode(entry); err != nil {
			_ = f.Close()
			t.Fatalf("encode legacy WAL: %v", err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close legacy WAL: %v", err)
	}

	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("recover legacy WAL: loaded=%v err=%v", loaded, err)
	}
	if recovered.next != 2 {
		t.Fatalf("legacy auto-ID high-water=%d, want 2", recovered.next)
	}
	id, err := recovered.Add([]float32{0, 0, 1}, "third", "", nil, "default", "default")
	if err != nil {
		t.Fatalf("add after legacy recovery: %v", err)
	}
	if id != "doc-2" {
		t.Fatalf("auto ID after legacy recovery=%q, want doc-2", id)
	}
}

func TestReplayWALRejectsUnknownOperationWithoutMutation(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "index.gob.wal")
	writeModernWALRecords(t, walPath, walEntry{
		Seq: 1, Op: "explode", ID: "id-1", Vec: []float32{1, 0, 0},
	})

	store := NewVectorStore(10, 3)
	store.walPath = walPath
	if err := replayWAL(store); err == nil || !strings.Contains(err.Error(), "unknown WAL operation") {
		t.Fatalf("expected unknown-operation error, got %v", err)
	}
	if store.Count != 0 || store.appliedWALSeq != 0 {
		t.Fatalf("unknown operation mutated store: count=%d high_water=%d", store.Count, store.appliedWALSeq)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("rejected WAL was not preserved: %v", err)
	}
}

func TestReplayWALRejectsTruncatedTailBeforeMutation(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "index.gob.wal")
	writeModernWALRecords(t, walPath, walEntry{
		Seq: 1, Op: "insert", ID: "id-1", Doc: "valid", Vec: []float32{1, 0, 0}, Coll: "default", Tenant: "default",
	})
	f, err := os.OpenFile(walPath, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open WAL tail: %v", err)
	}
	if _, err := f.WriteString("{\"Version\":1,\"Seq\":2"); err != nil {
		_ = f.Close()
		t.Fatalf("write truncated WAL tail: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close truncated WAL: %v", err)
	}

	store := NewVectorStore(10, 3)
	store.walPath = walPath
	if err := replayWAL(store); err == nil {
		t.Fatal("expected truncated WAL to fail")
	}
	if store.Count != 0 || store.appliedWALSeq != 0 {
		t.Fatalf("parse-first recovery was not transactional: count=%d high_water=%d", store.Count, store.appliedWALSeq)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("truncated WAL was not preserved: %v", err)
	}
}

func TestSnapshotSkipsAlreadyCheckpointedWAL(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	if _, err := writer.Add([]float32{1, 0, 0}, "once", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save snapshot: %v", err)
	}

	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("reload snapshot plus stale WAL: loaded=%v err=%v", loaded, err)
	}
	if recovered.Count != 1 || recovered.GetDoc(0) != "once" || recovered.appliedWALSeq != 1 {
		t.Fatalf("stale WAL duplicated state: count=%d doc=%q high_water=%d", recovered.Count, recovered.GetDoc(0), recovered.appliedWALSeq)
	}
	if _, err := os.Stat(walPath); !os.IsNotExist(err) {
		t.Fatalf("checkpointed stale WAL was not cleaned up: %v", err)
	}
}

func TestVersion3CanonicalSnapshotMigratesToWALHighWaterFormat(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")

	writer := NewVectorStore(10, 3)
	if _, err := writer.Add([]float32{1, 0, 0}, "version-three", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save current snapshot: %v", err)
	}
	payload, format, err := tryLoadPayload(snapshotPath)
	if err != nil {
		t.Fatalf("read current snapshot: %v", err)
	}
	payload.FormatVersion = 3
	payload.WALHighWater = 0
	payload.Checksum = writer.computeV3Checksum()
	f, err := os.OpenFile(snapshotPath, os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open version 3 snapshot: %v", err)
	}
	if err := format.Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("encode version 3 snapshot: %v", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		t.Fatalf("sync version 3 snapshot: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close version 3 snapshot: %v", err)
	}

	migrated, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("load canonical version 3 snapshot: loaded=%v err=%v", loaded, err)
	}
	if migrated.Count != 1 || migrated.GetDoc(0) != "version-three" || migrated.appliedWALSeq != 0 {
		t.Fatalf("version 3 migration changed state: count=%d doc=%q high_water=%d", migrated.Count, migrated.GetDoc(0), migrated.appliedWALSeq)
	}
	if !migrated.validateChecksum() {
		t.Fatal("version 3 state was not migrated to the current checksum")
	}
	if err := migrated.Save(snapshotPath); err != nil {
		t.Fatalf("rewrite migrated snapshot: %v", err)
	}
	rewritten, _, err := tryLoadPayload(snapshotPath)
	if err != nil {
		t.Fatalf("read rewritten snapshot: %v", err)
	}
	if rewritten.FormatVersion != storage.CurrentFormatVersion {
		t.Fatalf("rewritten snapshot format=%d, want %d", rewritten.FormatVersion, storage.CurrentFormatVersion)
	}
}

func TestVersion3SnapshotCannotDowngradeToWeakLegacyChecksum(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	writer := NewVectorStore(10, 3)
	if _, err := writer.Add([]float32{1, 0, 0}, "version-three", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save: %v", err)
	}
	payload, format, err := tryLoadPayload(snapshotPath)
	if err != nil {
		t.Fatalf("load payload: %v", err)
	}
	payload.FormatVersion = storage.StrictFormatVersion
	payload.WALHighWater = 0
	payload.Checksum = fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d-%d", payload.Count, payload.Next, len(payload.Docs))))
	f, err := os.OpenFile(snapshotPath, os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open weak version 3 snapshot: %v", err)
	}
	if err := format.Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("write weak version 3 snapshot: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close weak version 3 snapshot: %v", err)
	}
	if store, loaded, err := loadOrInitStore(snapshotPath, 10, 3); err == nil || store != nil || loaded {
		t.Fatalf("weak version 3 checksum was accepted: store=%v loaded=%v err=%v", store, loaded, err)
	}
}

func TestConcurrentSnapshotsCannotCommitStaleState(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	store := NewVectorStore(10, 3)
	if _, err := store.Add([]float32{1, 0, 0}, "first", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("seed first state: %v", err)
	}

	originalRename := renameSnapshotFile
	defer func() { renameSnapshotFile = originalRename }()
	firstRename := true
	firstCaptured := make(chan struct{})
	releaseFirst := make(chan struct{})
	renameSnapshotFile = func(oldPath, newPath string) error {
		if firstRename {
			firstRename = false
			close(firstCaptured)
			<-releaseFirst
		}
		return originalRename(oldPath, newPath)
	}

	firstResult := make(chan error, 1)
	go func() { firstResult <- store.Save(snapshotPath) }()
	select {
	case <-firstCaptured:
	case <-time.After(5 * time.Second):
		t.Fatal("first snapshot did not reach commit point")
	}

	// The first snapshot has released the store read lock but has not renamed
	// its older image. Install newer state and start a second save; snapshotMu
	// must prevent that newer save from committing ahead of the older one.
	if _, err := store.Add([]float32{0, 1, 0}, "second", "id-2", nil, "default", "default"); err != nil {
		t.Fatalf("install newer state: %v", err)
	}
	secondResult := make(chan error, 1)
	go func() { secondResult <- store.Save(snapshotPath) }()
	close(releaseFirst)
	if err := <-firstResult; err != nil {
		t.Fatalf("first save: %v", err)
	}
	if err := <-secondResult; err != nil {
		t.Fatalf("second save: %v", err)
	}

	renameSnapshotFile = originalRename
	loaded, ok, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !ok {
		t.Fatalf("load final snapshot: loaded=%v err=%v", ok, err)
	}
	if loaded.Count != 2 || loaded.GetDoc(0) != "first" || loaded.GetDoc(1) != "second" {
		t.Fatalf("stale snapshot committed last: count=%d docs=%v", loaded.Count, loaded.Docs)
	}
}

func TestWALRecoveryRejectsConflictingDuplicateInsert(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	if _, err := writer.Add([]float32{1, 0, 0}, "original", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := removeDurableFile(walPath); err != nil {
		t.Fatalf("remove checkpoint source WAL: %v", err)
	}
	writeModernWALRecords(t, walPath, walEntry{
		Seq: 2, Op: "insert", ID: "id-1", Doc: "conflict", Vec: []float32{0, 1, 0}, Coll: "default", Tenant: "default",
	})

	store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err == nil || store != nil || loaded || !strings.Contains(err.Error(), "conflicting replayed insert") {
		t.Fatalf("conflicting insert must fail closed: store=%v loaded=%v err=%v", store, loaded, err)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("conflicting WAL was not preserved: %v", err)
	}
}

func TestWALRecoveryReplaysFrozenThenCurrent(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	frozenPath := walPath + ".frozen"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	if _, err := writer.Add([]float32{1, 0, 0}, "original", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := removeDurableFile(walPath); err != nil {
		t.Fatalf("remove checkpoint source WAL: %v", err)
	}
	writeModernWALRecords(t, frozenPath, walEntry{
		Seq: 2, Op: "upsert", ID: "id-1", Doc: "middle", Vec: []float32{0, 1, 0}, Coll: "default", Tenant: "default",
	})
	writeModernWALRecords(t, walPath, walEntry{
		Seq: 3, Op: "upsert", ID: "id-1", Doc: "newest", Vec: []float32{0, 0, 1}, Coll: "default", Tenant: "default",
	})

	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("recover frozen/current WALs: loaded=%v err=%v", loaded, err)
	}
	if recovered.Count != 1 || recovered.GetDoc(0) != "newest" || recovered.appliedWALSeq != 3 {
		t.Fatalf("WAL order was not preserved: count=%d doc=%q high_water=%d", recovered.Count, recovered.GetDoc(0), recovered.appliedWALSeq)
	}
	if got := recovered.Get(0); len(got) != 3 || got[2] != 1 {
		t.Fatalf("latest vector was not recovered: %v", got)
	}
	for _, path := range []string{frozenPath, walPath} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("checkpointed WAL artifact %q remains: %v", path, err)
		}
	}
}

func TestWALRecoveryValidatesAllArtifactsBeforeApplying(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	frozenPath := walPath + ".frozen"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	if _, err := writer.Add([]float32{1, 0, 0}, "original", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := removeDurableFile(walPath); err != nil {
		t.Fatalf("remove checkpoint source WAL: %v", err)
	}
	writeModernWALRecords(t, frozenPath, walEntry{
		Seq: 2, Op: "upsert", ID: "id-1", Doc: "would-apply", Vec: []float32{0, 1, 0}, Coll: "default", Tenant: "default",
	})
	if err := os.WriteFile(walPath, []byte("{corrupt-current\n"), 0o600); err != nil {
		t.Fatalf("write corrupt current WAL: %v", err)
	}

	store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err == nil || store != nil || loaded {
		t.Fatalf("corrupt current WAL must fail before recovery: store=%v loaded=%v err=%v", store, loaded, err)
	}
	for _, path := range []string{frozenPath, walPath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("recovery failure removed %q: %v", path, err)
		}
	}
}

func TestWALRecoveryCleanupFailureIsRetryable(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"

	writer := NewVectorStore(10, 3)
	writer.walPath = walPath
	if _, err := writer.Add([]float32{1, 0, 0}, "original", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := writer.Save(snapshotPath); err != nil {
		t.Fatalf("save: %v", err)
	}
	writeModernWALRecords(t, walPath, walEntry{
		Seq: 2, Op: "upsert", ID: "id-1", Doc: "recovered", Vec: []float32{0, 1, 0}, Coll: "default", Tenant: "default",
	})

	originalRemove := removeWALArtifact
	defer func() { removeWALArtifact = originalRemove }()
	removeWALArtifact = func(string) error { return errors.New("injected cleanup failure") }
	store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err == nil || store != nil || loaded || !strings.Contains(err.Error(), "injected cleanup failure") {
		t.Fatalf("cleanup failure must surface: store=%v loaded=%v err=%v", store, loaded, err)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("cleanup failure did not preserve WAL: %v", err)
	}

	removeWALArtifact = originalRemove
	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("retry recovery after cleanup failure: loaded=%v err=%v", loaded, err)
	}
	if recovered.Count != 1 || recovered.GetDoc(0) != "recovered" || recovered.appliedWALSeq != 2 {
		t.Fatalf("retry duplicated or lost state: count=%d doc=%q high_water=%d", recovered.Count, recovered.GetDoc(0), recovered.appliedWALSeq)
	}
	if _, err := os.Stat(walPath); !os.IsNotExist(err) {
		t.Fatalf("retry did not remove stale WAL: %v", err)
	}
}

func TestWALChecksumTamperFailsClosed(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	entry := walEntry{
		Version:  currentWALVersion,
		Seq:      1,
		Op:       "insert",
		ID:       "id-1",
		Doc:      "tampered",
		Vec:      []float32{1, 0, 0},
		Coll:     "default",
		Tenant:   "default",
		Checksum: "sha256:" + strings.Repeat("0", 64),
	}
	f, err := os.OpenFile(walPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("create tampered WAL: %v", err)
	}
	if err := json.NewEncoder(f).Encode(entry); err != nil {
		_ = f.Close()
		t.Fatalf("encode tampered WAL: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close tampered WAL: %v", err)
	}

	store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err == nil || store != nil || loaded || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("tampered WAL must fail closed: store=%v loaded=%v err=%v", store, loaded, err)
	}
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("tampered WAL was not preserved: %v", err)
	}
}

func TestIndeterminateWALAppendLatchesWriteFault(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	store := NewVectorStore(10, 3)
	store.walPath = walPath

	originalSync := syncWALFile
	defer func() { syncWALFile = originalSync }()
	syncWALFile = func(*os.File) error { return errors.New("injected fsync uncertainty") }
	if _, err := store.Add([]float32{1, 0, 0}, "uncertain", "id-1", nil, "default", "default"); err == nil || !strings.Contains(err.Error(), "indeterminate") {
		t.Fatalf("expected indeterminate append error, got %v", err)
	}
	if store.Count != 0 || store.walFault == nil {
		t.Fatalf("failed append must roll back logical state and latch fault: count=%d fault=%v", store.Count, store.walFault)
	}

	syncWALFile = originalSync
	if _, err := store.Add([]float32{0, 1, 0}, "must-not-append", "id-2", nil, "default", "default"); err == nil || !strings.Contains(err.Error(), "writes are disabled") {
		t.Fatalf("latched WAL fault allowed another write: %v", err)
	}
	if store.Count != 0 {
		t.Fatalf("latched WAL fault mutated state: count=%d", store.Count)
	}

	// The first request had an indeterminate outcome. A restart validates and
	// replays its complete record, while the poisoned process never reused LSN 1.
	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("recover indeterminate append: loaded=%v err=%v", loaded, err)
	}
	if recovered.Count != 1 || recovered.GetID(0) != "id-1" || recovered.appliedWALSeq != 1 {
		t.Fatalf("indeterminate append recovery mismatch: count=%d id=%q high_water=%d", recovered.Count, recovered.GetID(0), recovered.appliedWALSeq)
	}
}

func TestDeleteRemainsVisibleWhenWALAppendFails(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	store := NewVectorStore(10, 3)
	store.walPath = walPath
	if _, err := store.Add([]float32{1, 0, 0}, "keep-until-durable", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("seed document: %v", err)
	}
	if err := store.Save(snapshotPath); err != nil {
		t.Fatalf("save seed snapshot: %v", err)
	}
	if err := removeDurableFile(walPath); err != nil {
		t.Fatalf("remove seed WAL: %v", err)
	}

	originalSync := syncWALFile
	defer func() { syncWALFile = originalSync }()
	syncWALFile = func(*os.File) error { return errors.New("injected delete fsync uncertainty") }
	if err := store.Delete("id-1"); err == nil || !strings.Contains(err.Error(), "indeterminate") {
		t.Fatalf("expected delete WAL failure, got %v", err)
	}
	if store.Deleted[hashID("id-1")] || store.GetDoc(0) != "keep-until-durable" {
		t.Fatalf("failed delete changed logical state: deleted=%v doc=%q", store.Deleted[hashID("id-1")], store.GetDoc(0))
	}
	results := store.SearchANN([]float32{1, 0, 0}, 1)
	if len(results) != 1 || store.GetID(results[0]) != "id-1" {
		t.Fatalf("failed delete removed index entry: results=%v", results)
	}

	// The fsynced outcome is indeterminate to the caller, so restart may finish
	// the delete; the poisoned process itself never exposes it prematurely.
	syncWALFile = originalSync
	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("recover indeterminate delete: loaded=%v err=%v", loaded, err)
	}
	if !recovered.Deleted[hashID("id-1")] {
		t.Fatal("restart did not replay complete indeterminate delete record")
	}
}

func TestIndeterminateWALRotationSyncLatchesWriteFault(t *testing.T) {
	dir := t.TempDir()
	snapshotPath := filepath.Join(dir, "index.gob")
	walPath := snapshotPath + ".wal"
	store := NewVectorStore(10, 3)
	store.walPath = walPath
	store.walMaxOps = 1

	originalSyncDir := syncWALDirectory
	defer func() { syncWALDirectory = originalSyncDir }()
	syncCalls := 0
	syncWALDirectory = func(path string) error {
		syncCalls++
		if syncCalls == 2 {
			return errors.New("injected rotation directory sync uncertainty")
		}
		return originalSyncDir(path)
	}
	if _, err := store.Add([]float32{1, 0, 0}, "uncertain-rotation", "id-1", nil, "default", "default"); err == nil || !strings.Contains(err.Error(), "indeterminate") {
		t.Fatalf("expected indeterminate rotation error, got %v", err)
	}
	if store.Count != 0 || store.walFault == nil {
		t.Fatalf("rotation uncertainty must roll back state and latch fault: count=%d fault=%v", store.Count, store.walFault)
	}
	if _, err := os.Stat(walPath + ".frozen"); err != nil {
		t.Fatalf("rotated recovery artifact missing: %v", err)
	}

	syncWALDirectory = originalSyncDir
	recovered, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
	if err != nil || !loaded {
		t.Fatalf("recover indeterminate rotation: loaded=%v err=%v", loaded, err)
	}
	if recovered.Count != 1 || recovered.GetID(0) != "id-1" {
		t.Fatalf("rotation recovery mismatch: count=%d id=%q", recovered.Count, recovered.GetID(0))
	}
}

func TestWALRecoveryRejectsSequenceGapAndMixedFormats(t *testing.T) {
	t.Run("sequence gap", func(t *testing.T) {
		dir := t.TempDir()
		snapshotPath := filepath.Join(dir, "index.gob")
		walPath := snapshotPath + ".wal"
		writeModernWALRecords(t, walPath, walEntry{
			Seq: 2, Op: "insert", ID: "id-2", Doc: "gap", Vec: []float32{1, 0, 0}, Coll: "default", Tenant: "default",
		})
		store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
		if err == nil || store != nil || loaded || !strings.Contains(err.Error(), "sequence gap") {
			t.Fatalf("sequence gap must fail closed: store=%v loaded=%v err=%v", store, loaded, err)
		}
		if _, err := os.Stat(walPath); err != nil {
			t.Fatalf("gapped WAL was not preserved: %v", err)
		}
	})

	t.Run("mixed legacy and sequenced", func(t *testing.T) {
		dir := t.TempDir()
		snapshotPath := filepath.Join(dir, "index.gob")
		frozenPath := snapshotPath + ".wal.frozen"
		walPath := snapshotPath + ".wal"
		legacy, err := os.OpenFile(frozenPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatalf("create legacy frozen WAL: %v", err)
		}
		if err := json.NewEncoder(legacy).Encode(walEntry{Op: "insert", ID: "legacy", Doc: "legacy", Vec: []float32{1, 0, 0}}); err != nil {
			_ = legacy.Close()
			t.Fatalf("encode legacy WAL: %v", err)
		}
		if err := legacy.Close(); err != nil {
			t.Fatalf("close legacy WAL: %v", err)
		}
		writeModernWALRecords(t, walPath, walEntry{
			Seq: 1, Op: "insert", ID: "current", Doc: "current", Vec: []float32{0, 1, 0}, Coll: "default", Tenant: "default",
		})
		store, loaded, err := loadOrInitStore(snapshotPath, 10, 3)
		if err == nil || store != nil || loaded || !strings.Contains(err.Error(), "mixed legacy") {
			t.Fatalf("mixed WAL formats must fail closed: store=%v loaded=%v err=%v", store, loaded, err)
		}
		for _, path := range []string{frozenPath, walPath} {
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("mixed-format recovery removed %q: %v", path, err)
			}
		}
	})
}
