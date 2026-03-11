package index

// GraphStore abstracts the neighbor graph storage for DiskANN.
// This enables swapping between in-memory and disk-backed (mmap) implementations.
type GraphStore interface {
	// GetNeighbors returns the neighbor list for a node. Returns nil if not found.
	GetNeighbors(id uint64) []uint64
	// SetNeighbors replaces the neighbor list for a node.
	SetNeighbors(id uint64, neighbors []uint64)
	// HasNode returns true if the node exists in the graph.
	HasNode(id uint64) bool
	// DeleteNode removes a node from the graph.
	DeleteNode(id uint64)
	// Len returns the number of nodes in the graph.
	Len() int
	// Range iterates over all nodes. Return false from fn to stop.
	Range(fn func(id uint64, neighbors []uint64) bool)
	// Snapshot returns a read-only shallow copy for lock-free search.
	Snapshot() GraphStore
	// Clone returns a deep copy (for export/serialization).
	Clone() map[uint64][]uint64
	// ReplaceAll replaces the entire graph contents from a map.
	ReplaceAll(graph map[uint64][]uint64)
}

// MemoryGraphStore is an in-memory GraphStore backed by a Go map.
// This is the default implementation and has identical behavior to the
// original map[uint64][]uint64 field.
type MemoryGraphStore struct {
	graph map[uint64][]uint64
}

// NewMemoryGraphStore creates a new in-memory graph store.
func NewMemoryGraphStore() *MemoryGraphStore {
	return &MemoryGraphStore{
		graph: make(map[uint64][]uint64),
	}
}

// NewMemoryGraphStoreFrom creates a MemoryGraphStore from an existing map (takes ownership).
func NewMemoryGraphStoreFrom(graph map[uint64][]uint64) *MemoryGraphStore {
	return &MemoryGraphStore{graph: graph}
}

func (m *MemoryGraphStore) GetNeighbors(id uint64) []uint64 {
	return m.graph[id]
}

func (m *MemoryGraphStore) SetNeighbors(id uint64, neighbors []uint64) {
	m.graph[id] = neighbors
}

func (m *MemoryGraphStore) HasNode(id uint64) bool {
	_, ok := m.graph[id]
	return ok
}

func (m *MemoryGraphStore) DeleteNode(id uint64) {
	delete(m.graph, id)
}

func (m *MemoryGraphStore) Len() int {
	return len(m.graph)
}

func (m *MemoryGraphStore) Range(fn func(id uint64, neighbors []uint64) bool) {
	for id, neighbors := range m.graph {
		if !fn(id, neighbors) {
			return
		}
	}
}

func (m *MemoryGraphStore) Snapshot() GraphStore {
	return NewMemoryGraphStoreFrom(m.Clone())
}

func (m *MemoryGraphStore) Clone() map[uint64][]uint64 {
	dst := make(map[uint64][]uint64, len(m.graph))
	for id, neighbors := range m.graph {
		dst[id] = append([]uint64(nil), neighbors...)
	}
	return dst
}

func (m *MemoryGraphStore) ReplaceAll(graph map[uint64][]uint64) {
	m.graph = graph
}

// FirstNodeID returns the first non-deleted node ID for use as entry point.
// deleted is optional (can be nil).
func FirstNodeID(gs GraphStore, deleted map[uint64]bool) (uint64, bool) {
	var entryID uint64
	found := false
	gs.Range(func(id uint64, _ []uint64) bool {
		if !deleted[id] {
			entryID = id
			found = true
			return false
		}
		return true
	})
	return entryID, found
}
