package hnsw

import (
	"cmp"
	"sync"
	"unsafe"
)

// nodeLockPool provides fine-grained locking for neighbor list mutations.
// Uses pointer-address sharding to avoid needing K to be hashable.
// 256 shards gives <1% contention for 8-16 concurrent inserters.
type nodeLockPool[K cmp.Ordered] struct {
	shards [256]sync.Mutex
}

func (p *nodeLockPool[K]) shard(n *layerNode[K]) *sync.Mutex {
	// Hash the pointer address to a shard index.
	// Shift right by 6 to skip alignment bits (layerNode is >64 bytes).
	idx := uint(uintptr(unsafe.Pointer(n)) >> 6) % 256
	return &p.shards[idx]
}

func (p *nodeLockPool[K]) lock(n *layerNode[K]) {
	p.shard(n).Lock()
}

func (p *nodeLockPool[K]) unlock(n *layerNode[K]) {
	p.shard(n).Unlock()
}
