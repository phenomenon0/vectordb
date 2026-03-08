package cluster

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/testutil"
)

type mockStore struct {
	mu sync.RWMutex

	dim       int
	ids       []string
	docs      []string
	data      []float32
	seqs      []uint64
	meta      map[uint64]map[string]string
	deleted   map[uint64]bool
	coll      map[uint64]string
	idToIx    map[uint64]int
	apiToken  string
	authReq   bool
	jwt       JWTValidator
	walHook   func(WalEntry)
	addErr    error
	upsertErr error

	addCalls    int
	upsertCalls int
	deleteCalls int
}

func newMockStore(dim int) *mockStore {
	return &mockStore{
		dim:     dim,
		meta:    make(map[uint64]map[string]string),
		deleted: make(map[uint64]bool),
		coll:    make(map[uint64]string),
		idToIx:  make(map[uint64]int),
	}
}

func (m *mockStore) StoreLock()   { m.mu.Lock() }
func (m *mockStore) StoreUnlock() { m.mu.Unlock() }
func (m *mockStore) StoreRLock()  { m.mu.RLock() }
func (m *mockStore) StoreRUnlock() {
	m.mu.RUnlock()
}

func (m *mockStore) StoreCount() int                         { return len(m.ids) }
func (m *mockStore) StoreDim() int                           { return m.dim }
func (m *mockStore) StoreIDs() []string                      { return append([]string(nil), m.ids...) }
func (m *mockStore) StoreDocs() []string                     { return append([]string(nil), m.docs...) }
func (m *mockStore) StoreData() []float32                    { return append([]float32(nil), m.data...) }
func (m *mockStore) StoreSeqs() []uint64                     { return append([]uint64(nil), m.seqs...) }
func (m *mockStore) StoreMeta() map[uint64]map[string]string { return m.meta }
func (m *mockStore) StoreDeleted() map[uint64]bool           { return m.deleted }
func (m *mockStore) StoreColl() map[uint64]string            { return m.coll }
func (m *mockStore) StoreIDToIx() map[uint64]int             { return m.idToIx }
func (m *mockStore) StoreAPIToken() string                   { return m.apiToken }
func (m *mockStore) StoreRequireAuth() bool                  { return m.authReq }
func (m *mockStore) StoreJWTMgr() JWTValidator               { return m.jwt }

func (m *mockStore) Add(vec []float32, doc string, id string, meta map[string]string, coll string, tenant string) (string, error) {
	if m.addErr != nil {
		return "", m.addErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	m.addCalls++
	if id == "" {
		id = fmt.Sprintf("doc-%d", len(m.ids)+1)
	}
	if coll == "" {
		coll = "default"
	}
	hid := testHashID(id)
	if _, exists := m.idToIx[hid]; exists {
		return "", fmt.Errorf("duplicate id %s", id)
	}

	m.idToIx[hid] = len(m.ids)
	m.ids = append(m.ids, id)
	m.docs = append(m.docs, doc)
	m.data = append(m.data, vec...)
	m.seqs = append(m.seqs, uint64(len(m.seqs)))
	m.coll[hid] = coll
	if meta != nil {
		cp := make(map[string]string, len(meta))
		for k, v := range meta {
			cp[k] = v
		}
		m.meta[hid] = cp
	}
	delete(m.deleted, hid)
	if m.walHook != nil {
		m.walHook(WalEntry{Op: "insert", ID: id, Doc: doc, Meta: meta, Vec: vec, Coll: coll, Tenant: tenant})
	}
	return id, nil
}

func (m *mockStore) Upsert(vec []float32, doc string, id string, meta map[string]string, coll string, tenant string) (string, error) {
	if m.upsertErr != nil {
		return "", m.upsertErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	m.upsertCalls++
	if id == "" {
		id = fmt.Sprintf("doc-%d", len(m.ids)+1)
	}
	if coll == "" {
		coll = "default"
	}
	hid := testHashID(id)
	if ix, exists := m.idToIx[hid]; exists {
		copy(m.data[ix*m.dim:(ix+1)*m.dim], vec)
		m.docs[ix] = doc
		m.coll[hid] = coll
		if meta != nil {
			cp := make(map[string]string, len(meta))
			for k, v := range meta {
				cp[k] = v
			}
			m.meta[hid] = cp
		}
		delete(m.deleted, hid)
	} else {
		m.idToIx[hid] = len(m.ids)
		m.ids = append(m.ids, id)
		m.docs = append(m.docs, doc)
		m.data = append(m.data, vec...)
		m.seqs = append(m.seqs, uint64(len(m.seqs)))
		m.coll[hid] = coll
		if meta != nil {
			cp := make(map[string]string, len(meta))
			for k, v := range meta {
				cp[k] = v
			}
			m.meta[hid] = cp
		}
	}
	delete(m.deleted, hid)
	return id, nil
}

func (m *mockStore) DeleteByID(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deleteCalls++
	m.deleted[testHashID(id)] = true
}

func (m *mockStore) Save(path string) error { return nil }

func (m *mockStore) SetAPIToken(token string) {
	m.apiToken = token
}

func (m *mockStore) SetWALHook(h func(WalEntry)) {
	m.walHook = h
}

func (m *mockStore) LoadSnapshotData(snap *SnapshotData) error {
	m.dim = snap.Dim
	m.ids = append([]string(nil), snap.IDs...)
	m.docs = append([]string(nil), snap.Docs...)
	m.data = append([]float32(nil), snap.Data...)
	m.seqs = append([]uint64(nil), snap.Seqs...)
	m.meta = snap.Meta
	if m.meta == nil {
		m.meta = make(map[uint64]map[string]string)
	}
	m.deleted = snap.Deleted
	if m.deleted == nil {
		m.deleted = make(map[uint64]bool)
	}
	m.coll = snap.Coll
	if m.coll == nil {
		m.coll = make(map[uint64]string)
	}
	m.idToIx = snap.IDToIx
	if m.idToIx == nil {
		m.idToIx = make(map[uint64]int)
	}
	return nil
}

func (m *mockStore) CreateSnapshotData() *SnapshotData {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return &SnapshotData{
		Count:   len(m.ids),
		Dim:     m.dim,
		IDs:     append([]string(nil), m.ids...),
		Docs:    append([]string(nil), m.docs...),
		Data:    append([]float32(nil), m.data...),
		Seqs:    append([]uint64(nil), m.seqs...),
		Meta:    m.meta,
		Deleted: m.deleted,
		Coll:    m.coll,
		IDToIx:  m.idToIx,
	}
}

func testHashID(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func TestHandleSnapshotRequiresAuth(t *testing.T) {
	store := newMockStore(2)
	store.apiToken = "secret"
	_, err := store.Add([]float32{1, 2}, "doc", "doc-1", nil, "default", "")
	if err != nil {
		t.Fatalf("seed add failed: %v", err)
	}

	shard := &ShardServer{
		role:  RolePrimary,
		store: store,
		deps: &Deps{
			HashID: testHashID,
		},
	}

	req := httptest.NewRequest(http.MethodGet, "/internal/snapshot", nil)
	w := httptest.NewRecorder()
	shard.handleSnapshot(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without auth, got %d", w.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/internal/snapshot", nil)
	req.Header.Set("Authorization", "Bearer secret")
	w = httptest.NewRecorder()
	shard.handleSnapshot(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with auth, got %d", w.Code)
	}
}

func TestApplyEntriesUsesUpsertForUpserts(t *testing.T) {
	store := newMockStore(2)
	shard := &ShardServer{store: store}

	entries := []WalEntry{
		{Seq: 1, Op: "insert", ID: "doc-1", Doc: "one", Vec: []float32{1, 2}, Coll: "default"},
		{Seq: 2, Op: "upsert", ID: "doc-1", Doc: "two", Vec: []float32{3, 4}, Coll: "default"},
	}
	if err := shard.ApplyEntries(entries); err != nil {
		t.Fatalf("ApplyEntries failed: %v", err)
	}
	if store.addCalls != 1 {
		t.Fatalf("expected 1 add call, got %d", store.addCalls)
	}
	if store.upsertCalls != 1 {
		t.Fatalf("expected 1 upsert call, got %d", store.upsertCalls)
	}
}

func TestFollowerPollAppliesAllReturnedEntriesAcrossBatches(t *testing.T) {
	store := newMockStore(2)
	server := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("since"); got != "0" {
			t.Fatalf("expected first poll to request since=0, got %s", got)
		}
		resp := WALStreamResponse{
			Entries: []WalEntry{
				{Seq: 1, Op: "insert", ID: "doc-1", Doc: "one", Vec: []float32{1, 2}, Coll: "default"},
				{Seq: 2, Op: "insert", ID: "doc-2", Doc: "two", Vec: []float32{3, 4}, Coll: "default"},
			},
			LatestSeq: 2,
			Count:     2,
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode failed: %v", err)
		}
	}))

	replicator := NewFollowerReplicator(&ShardServer{store: store}, FollowerReplicatorConfig{
		PrimaryAddr:          server.URL,
		PollInterval:         time.Millisecond,
		ReconnectInterval:    time.Millisecond,
		MaxReconnectAttempts: 1,
		BatchSize:            1,
	})
	if err := replicator.poll(); err != nil {
		t.Fatalf("poll failed: %v", err)
	}

	if got := store.StoreCount(); got != 2 {
		t.Fatalf("expected 2 applied entries, got %d", got)
	}
	if replicator.lastAppliedSeq != 2 {
		t.Fatalf("expected lastAppliedSeq=2, got %d", replicator.lastAppliedSeq)
	}
	if replicator.walClient.lastSeq != 2 {
		t.Fatalf("expected wal client ack at seq 2, got %d", replicator.walClient.lastSeq)
	}
}

func TestRecoverFromWALReturnsHandlerError(t *testing.T) {
	dir := t.TempDir()
	adapter, err := NewWALAdapter(dir)
	if err != nil {
		t.Fatalf("NewWALAdapter failed: %v", err)
	}
	if err := adapter.Insert("doc-1", "doc", "default", "", []float32{1, 2}, nil); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	if err := adapter.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	store := newMockStore(2)
	store.addErr = errors.New("boom")
	err = RecoverFromWAL(store, dir)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected replay error containing boom, got %v", err)
	}
}

func TestRecoverFromWALFailsForCollectionOperations(t *testing.T) {
	dir := t.TempDir()
	adapter, err := NewWALAdapter(dir)
	if err != nil {
		t.Fatalf("NewWALAdapter failed: %v", err)
	}
	if err := adapter.CreateCollection("new-coll", ""); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	if err := adapter.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	err = RecoverFromWAL(newMockStore(2), dir)
	if !errors.Is(err, ErrCollectionReplayUnsupported) {
		t.Fatalf("expected ErrCollectionReplayUnsupported, got %v", err)
	}
}

func TestRequestFullSyncUpdatesLagStateWithoutUnderflow(t *testing.T) {
	snapshot := &Snapshot{
		ShardID:   1,
		Sequence:  42,
		Timestamp: time.Now(),
		Vectors: []SnapshotEntry{
			{ID: "doc-1", Vector: []float32{1, 2}},
		},
	}
	snapshot.Checksum = snapshot.computeChecksum()

	server := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Fatalf("expected auth header, got %q", got)
		}
		if err := json.NewEncoder(w).Encode(snapshot); err != nil {
			t.Fatalf("encode failed: %v", err)
		}
	}))

	replicator := NewFollowerReplicator(&ShardServer{store: newMockStore(2)}, FollowerReplicatorConfig{
		PrimaryAddr: server.URL,
		AuthToken:   "secret",
	})
	if err := replicator.RequestFullSync(context.Background()); err != nil {
		t.Fatalf("RequestFullSync failed: %v", err)
	}
	if replicator.lastAppliedSeq != snapshot.Sequence {
		t.Fatalf("expected lastAppliedSeq=%d, got %d", snapshot.Sequence, replicator.lastAppliedSeq)
	}
	if replicator.primarySeq != snapshot.Sequence {
		t.Fatalf("expected primarySeq=%d, got %d", snapshot.Sequence, replicator.primarySeq)
	}
	if replicator.walClient.lastSeq != snapshot.Sequence {
		t.Fatalf("expected WAL client seq=%d, got %d", snapshot.Sequence, replicator.walClient.lastSeq)
	}

	replicator.mu.Lock()
	replicator.status = StatusStreaming
	replicator.stats.LastPollTime = time.Now()
	replicator.mu.Unlock()

	stats := replicator.GetStats()
	if lag, ok := stats["lag"].(uint64); !ok || lag != 0 {
		t.Fatalf("expected lag=0, got %#v", stats["lag"])
	}
	if !replicator.FollowerHealthCheck() {
		t.Fatal("expected follower health check to stay healthy after full sync")
	}
}
