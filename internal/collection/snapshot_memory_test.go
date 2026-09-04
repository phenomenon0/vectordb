package collection

import (
	"context"
	"encoding/binary"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// RCV-03: a unified v2 snapshot must load with heap bounded by one frame plus
// the live state, never by the file. These tests measure live heap after
// runtime.GC() with the collector otherwise disabled, so a pass or fail does
// not depend on GC timing.
//
// What they do NOT prove: peak in-flight heap under a hard cap. A transient
// whole-body read that is dropped before return passes the retention checks;
// only the manager-reader test below sees such a read, and only for the
// manager frames. The cap rehearsal under a cgroup is gate RCV-05.

const (
	snapshotMemoryDim       = 512
	snapshotMemoryDocuments = 2048
)

// writeLargeV2SnapshotForMemory commits a multi-MiB snapshot with one frame
// per document and returns nothing that keeps the source manager alive.
// Vector values are full-precision float32 so their JSON text (~11 bytes per
// value) is wider than the two in-memory copies (document + FLAT index, 8
// bytes per value) that a load rebuilds.
func writeLargeV2SnapshotForMemory(t *testing.T) (string, int64) {
	t.Helper()
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := NewCollectionManager(basePath)
	if _, err := manager.CreateCollection(context.Background(), testSchema("bulk", snapshotMemoryDim)); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	ids := make([]uint64, snapshotMemoryDocuments)
	vectors := make([][]float32, snapshotMemoryDocuments)
	for i := range ids {
		ids[i] = uint64(i + 1)
		vec := make([]float32, snapshotMemoryDim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}
	if err := manager.BulkAddDense(context.Background(), "bulk", "embedding", ids, vectors); err != nil {
		t.Fatal(err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	manager.closeAll()
	info, err := os.Stat(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() < 4<<20 {
		t.Fatalf("fixture snapshot is only %d bytes; the assertions need a multi-MiB file", info.Size())
	}
	return basePath, info.Size()
}

func disableGCForMeasurement(t *testing.T) {
	t.Helper()
	previous := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(previous) })
}

func liveHeapBytes() int64 {
	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return int64(stats.HeapAlloc)
}

// The validation pass exists so that a malformed file is rejected before any
// live state is built. If it started retaining frames (for example to hand
// them to the build pass instead of re-reading the file), the whole body
// would be resident alongside the rebuilt state.
func TestUnifiedCollectionSnapshotMemoryValidationRetainsNothing(t *testing.T) {
	basePath, fileSize := writeLargeV2SnapshotForMemory(t)
	f, version, info, err := openCollectionSnapshotFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if version != collectionSnapshotVersion {
		t.Fatalf("fixture version = %d, want %d", version, collectionSnapshotVersion)
	}
	disableGCForMeasurement(t)

	before := liveHeapBytes()
	validated, err := readUnifiedCollectionSnapshotV2File(f, info, "", false)
	if err != nil {
		t.Fatalf("validate framed snapshot: %v", err)
	}
	if validated.manager != nil || validated.tenants != nil {
		t.Fatal("validation pass built live managers")
	}
	retained := liveHeapBytes() - before
	runtime.KeepAlive(validated)
	// 1 MiB of slack absorbs runtime noise; a retained body is at least fileSize.
	if retained > 1<<20 {
		t.Fatalf("validation pass retained %d bytes of a %d byte snapshot; expected ~nothing", retained, fileSize)
	}
}

// A load rebuilds float32 vectors and indexes from frames; the file text must
// not survive behind that state. Retained heap below the file size fails for
// any loader that keeps the body (or a copy of it) reachable from the result.
func TestUnifiedCollectionSnapshotMemoryLoadRetainsLessThanFile(t *testing.T) {
	basePath, fileSize := writeLargeV2SnapshotForMemory(t)
	disableGCForMeasurement(t)

	before := liveHeapBytes()
	manager, tenants, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("open framed snapshot: %v", err)
	}
	retained := liveHeapBytes() - before
	if coll, err := manager.GetCollection("bulk"); err != nil || coll.Count() != snapshotMemoryDocuments {
		t.Fatalf("loaded collection = %v, %v; want %d documents", coll, err, snapshotMemoryDocuments)
	}
	runtime.KeepAlive(manager)
	runtime.KeepAlive(tenants)
	// Lower bound proves the measurement saw the rebuilt vectors at all.
	oneVectorCopy := int64(snapshotMemoryDocuments * snapshotMemoryDim * 4)
	if retained < oneVectorCopy {
		t.Fatalf("load retained %d bytes, less than one copy of the vectors (%d); measurement is not observing live state", retained, oneVectorCopy)
	}
	if retained >= fileSize {
		t.Fatalf("load retained %d bytes for a %d byte snapshot; the file text is being kept alive", retained, fileSize)
	}
}

type maxReadRecorder struct {
	r       io.Reader
	maxRead int
}

func (m *maxReadRecorder) Read(p []byte) (int, error) {
	if len(p) > m.maxRead {
		m.maxRead = len(p)
	}
	return m.r.Read(p)
}

// Frame-at-a-time is observable at the reader: no single Read may ask for
// more than one frame. A loader that slurps the remaining body into one
// buffer (io.ReadAll, ReadFull into a file-sized slice) requests a read of
// megabytes and fails here even though it drops the buffer before returning.
func TestUnifiedCollectionSnapshotMemoryManagerReadsOneFrameAtATime(t *testing.T) {
	basePath, fileSize := writeLargeV2SnapshotForMemory(t)
	f, err := os.Open(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	header := make([]byte, collectionSnapshotV2HeaderSize)
	if _, err := io.ReadFull(f, header); err != nil {
		t.Fatal(err)
	}
	_, rootCollectionCount, tenantCount, err := decodeCollectionSnapshotV2Header(header)
	if err != nil {
		t.Fatal(err)
	}
	if tenantCount != 0 {
		t.Fatalf("fixture has %d tenants; the body must be root-manager frames only", tenantCount)
	}
	bodySize := fileSize - int64(collectionSnapshotV2HeaderSize) - 32
	for _, build := range []bool{false, true} {
		if _, err := f.Seek(int64(collectionSnapshotV2HeaderSize), io.SeekStart); err != nil {
			t.Fatal(err)
		}
		recorder := &maxReadRecorder{r: io.LimitReader(f, bodySize)}
		manager, err := readCollectionSnapshotV2Manager(recorder, rootCollectionCount, "", build)
		if err != nil {
			t.Fatalf("read manager (build=%v): %v", build, err)
		}
		if build {
			manager.closeAll()
		}
		// Frames here are ~6 KiB; 64 KiB is far above any frame and far below the body.
		if recorder.maxRead > 64<<10 {
			t.Fatalf("build=%v: largest single read was %d bytes of a %d byte body; frames are not being read one at a time", build, recorder.maxRead, bodySize)
		}
	}
}

// The frame length prefix is untrusted input. Rejecting it must happen before
// make([]byte, size), otherwise a corrupt or hostile 4 GiB prefix allocates
// before it is refused. TotalAlloc is cumulative, so this is GC-independent.
func TestUnifiedCollectionSnapshotMemoryOversizeFrameRejectedBeforeAllocation(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "docs"), NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	path := collectionSnapshotPath(basePath)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	offset := int(collectionSnapshotV2HeaderSize)
	binary.BigEndian.PutUint32(data[offset:offset+4], collectionSnapshotV2MaxFrameSize+1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	_, _, _, err = OpenUnifiedCollectionSnapshot(basePath, basePath)
	runtime.ReadMemStats(&after)
	if err == nil || !strings.Contains(err.Error(), "maximum is") {
		t.Fatalf("expected oversize frame rejection, got %v", err)
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated >= uint64(collectionSnapshotV2MaxFrameSize) {
		t.Fatalf("rejecting an oversize frame prefix allocated %d bytes; the size check must run before the frame buffer exists", allocated)
	}
}
