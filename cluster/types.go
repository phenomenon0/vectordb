package cluster

import (
	"net/http"
	"time"
)

// ===========================================================================================
// DEPENDENCY INTERFACES
// These interfaces define what the cluster package needs from the main package.
// The main package's concrete types implement these interfaces.
// ===========================================================================================

// WalEntry mirrors the walEntry struct from the main package.
// Cluster code uses this for WAL streaming and replication.
type WalEntry struct {
	Seq    uint64
	Op     string
	ID     string
	Doc    string
	Meta   map[string]string
	Vec    []float32
	Coll   string
	Tenant string
	Time   int64
}

// Embedder is the interface cluster code needs for embedding text.
type Embedder interface {
	Embed(text string) ([]float32, error)
	Dim() int
}

// Reranker is the interface cluster code needs for reranking results.
type Reranker interface {
	Rerank(query string, docs []string, topK int) ([]int, []float64, error)
}

// JWTValidator is the interface cluster code needs for JWT validation in WAL auth.
type JWTValidator interface {
	ValidateTenantToken(token string) (*TenantContext, error)
}

// TenantContext represents a validated tenant context from JWT.
type TenantContext struct {
	TenantID string
	IsAdmin  bool
}

// MetricsProvider is the interface cluster code needs from MetricsCollector.
type MetricsProvider interface {
	RecordOperation(op string, shardID int, duration time.Duration, err error)
	Handler() http.Handler
	HTTPMiddleware(next http.Handler) http.Handler
	RecordReplicateError(operation string, shardID int, reason string)
}

// AuthProvider is the interface cluster code needs from AuthMiddleware.
type AuthProvider interface {
	Middleware(permissions ...string) func(http.HandlerFunc) http.HandlerFunc
}

// ===========================================================================================
// STORE INTERFACE
// Store abstracts the VectorStore for cluster code. The main package implements this
// on *VectorStore. Methods are named to avoid conflicts with sync.RWMutex embedding.
// ===========================================================================================

// Store is the interface that cluster code needs from VectorStore.
type Store interface {
	// Embedding sync.RWMutex equivalent
	StoreLock()
	StoreUnlock()
	StoreRLock()
	StoreRUnlock()

	// Read accessors
	StoreCount() int
	StoreDim() int
	StoreIDs() []string
	StoreDocs() []string
	StoreData() []float32
	StoreSeqs() []uint64
	StoreMeta() map[uint64]map[string]string
	StoreDeleted() map[uint64]bool
	StoreColl() map[uint64]string
	StoreIDToIx() map[uint64]int

	// Auth accessors (for WAL stream authorization)
	StoreAPIToken() string
	StoreRequireAuth() bool
	StoreJWTMgr() JWTValidator

	// Mutating operations
	Add(vec []float32, doc string, id string, meta map[string]string, coll string, tenant string) (string, error)
	Upsert(vec []float32, doc string, id string, meta map[string]string, coll string, tenant string) (string, error)
	DeleteByID(id string)
	Save(path string) error
	SetAPIToken(token string)
	SetWALHook(h func(WalEntry))

	// Snapshot support: bulk replace store contents
	// snapshot is the internal StoreSnapshot data
	LoadSnapshotData(snap *SnapshotData) error
	CreateSnapshotData() *SnapshotData
}

// SnapshotData holds the full store state for snapshot transfer.
// This avoids cluster code needing to know about every VectorStore field.
type SnapshotData struct {
	Count    int
	Dim      int
	IDs      []string
	Docs     []string
	Data     []float32
	Seqs     []uint64
	Meta     map[uint64]map[string]string
	NumMeta  map[uint64]map[string]float64
	TimeMeta map[uint64]map[string]time.Time
	Deleted  map[uint64]bool
	Coll     map[uint64]string
	TenantID map[uint64]string
	IDToIx   map[uint64]int
	LexTF    map[uint64]map[string]int
	DocLen   map[uint64]int
	DF       map[string]int
	SumDocL  int
}

// ===========================================================================================
// FACTORY FUNCTIONS / DEPENDENCY INJECTION
// ===========================================================================================

// Deps holds all dependencies that cluster code needs from the main package.
// Pass this when creating ShardServer or CoordinatorServer.
type Deps struct {
	// LoadOrInitStore loads or creates a new store.
	// Returns the Store and whether it was loaded from disk.
	LoadOrInitStore func(path string, capacity int, dim int) (Store, bool)

	// NewHTTPHandler creates the base HTTP handler for a store.
	NewHTTPHandler func(store Store, embedder Embedder, reranker Reranker, indexPath string) http.Handler

	// RegisterMigrationHandlers registers migration endpoints on a mux.
	RegisterMigrationHandlers func(mux *http.ServeMux, store Store)

	// HashID computes a hash for an ID string.
	HashID func(s string) uint64

	// NewEmbedder creates a default embedder for the given dimension.
	NewEmbedder func(dim int) Embedder

	// NewReranker creates a default reranker for the given embedder.
	NewReranker func(embedder Embedder) Reranker

	// NewMetrics creates a new metrics provider (nil if metrics disabled).
	NewMetrics func() MetricsProvider

	// NewAuth creates an auth middleware provider.
	NewAuth func(apiKeyMgr interface{}, jwtMgr interface{}) AuthProvider
}
