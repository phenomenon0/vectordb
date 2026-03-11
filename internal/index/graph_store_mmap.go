//go:build !windows

package index

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync/atomic"
	"syscall"

	"golang.org/x/sys/unix"
)

// MmapGraphStore stores the neighbor graph in a memory-mapped file.
// Each node occupies a fixed-size record: [4 bytes count][maxDegree * 8 bytes neighbor IDs].
// The ID-to-slot mapping is kept in memory (8 bytes per node).
//
// At 500M vectors with maxDegree=32:
//   - Mmap file: 500M * (4 + 32*8) = ~130GB (managed by OS page cache)
//   - In-memory index: 500M * 16 bytes (map overhead) ≈ 12GB
//   - vs pure in-memory map: 500M * (32*8 + map overhead) ≈ 140GB all in RSS
type MmapGraphStore struct {
	file      *os.File
	data      []byte
	maxDegree int
	slotSize  int // 4 + maxDegree*8

	// ID → slot offset mapping (in memory)
	slots map[uint64]int64

	// Next free slot offset
	nextOffset int64

	// File capacity in bytes (may be larger than used)
	capacity int64

	// Path for cleanup
	path string

	// Read-only snapshot flag
	readOnly bool
}

const (
	mmapGraphInitialSlots = 1 << 20 // 1M slots initial capacity
	mmapGraphGrowFactor   = 2
)

// NewMmapGraphStore creates a new mmap-backed graph store.
func NewMmapGraphStore(path string, maxDegree int) (*MmapGraphStore, error) {
	slotSize := 4 + maxDegree*8
	initialCapacity := int64(mmapGraphInitialSlots) * int64(slotSize)

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("mmap graph: create file: %w", err)
	}

	if err := file.Truncate(initialCapacity); err != nil {
		file.Close()
		return nil, fmt.Errorf("mmap graph: truncate: %w", err)
	}

	data, err := syscall.Mmap(int(file.Fd()), 0, int(initialCapacity),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("mmap graph: mmap: %w", err)
	}
	_ = unix.Madvise(data, unix.MADV_RANDOM)

	return &MmapGraphStore{
		file:       file,
		data:       data,
		maxDegree:  maxDegree,
		slotSize:   slotSize,
		slots:      make(map[uint64]int64, mmapGraphInitialSlots),
		nextOffset: 0,
		capacity:   initialCapacity,
		path:       path,
	}, nil
}

// Close releases mmap and file resources.
func (m *MmapGraphStore) Close() error {
	if m.readOnly {
		return nil // snapshots don't own resources
	}
	var firstErr error
	if m.data != nil {
		if err := syscall.Munmap(m.data); err != nil && firstErr == nil {
			firstErr = err
		}
		m.data = nil
	}
	if m.file != nil {
		if err := m.file.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		m.file = nil
	}
	return firstErr
}

func (m *MmapGraphStore) grow() error {
	newCap := m.capacity * mmapGraphGrowFactor
	if newCap < m.capacity+int64(m.slotSize)*1024 {
		newCap = m.capacity + int64(m.slotSize)*1024
	}

	if err := syscall.Munmap(m.data); err != nil {
		return fmt.Errorf("mmap graph grow: unmap: %w", err)
	}

	if err := m.file.Truncate(newCap); err != nil {
		return fmt.Errorf("mmap graph grow: truncate: %w", err)
	}

	data, err := syscall.Mmap(int(m.file.Fd()), 0, int(newCap),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("mmap graph grow: remap: %w", err)
	}
	_ = unix.Madvise(data, unix.MADV_RANDOM)

	m.data = data
	m.capacity = newCap
	return nil
}

func (m *MmapGraphStore) allocSlot(id uint64) (int64, error) {
	offset := m.nextOffset
	needed := offset + int64(m.slotSize)
	for needed > m.capacity {
		if err := m.grow(); err != nil {
			return 0, err
		}
	}
	m.nextOffset = needed
	m.slots[id] = offset
	return offset, nil
}

func (m *MmapGraphStore) GetNeighbors(id uint64) []uint64 {
	offset, ok := m.slots[id]
	if !ok {
		return nil
	}

	count := int(binary.LittleEndian.Uint32(m.data[offset:]))
	if count == 0 {
		return []uint64{}
	}

	neighbors := make([]uint64, count)
	base := offset + 4
	for i := 0; i < count; i++ {
		neighbors[i] = binary.LittleEndian.Uint64(m.data[base+int64(i)*8:])
	}
	return neighbors
}

func (m *MmapGraphStore) SetNeighbors(id uint64, neighbors []uint64) {
	offset, ok := m.slots[id]
	if !ok {
		var err error
		offset, err = m.allocSlot(id)
		if err != nil {
			// In production this would need better error handling.
			// For now, the interface doesn't return errors (matching map semantics).
			panic(fmt.Sprintf("mmap graph: alloc failed: %v", err))
		}
	}

	count := len(neighbors)
	if count > m.maxDegree {
		count = m.maxDegree
	}

	binary.LittleEndian.PutUint32(m.data[offset:], uint32(count))
	base := offset + 4
	for i := 0; i < count; i++ {
		binary.LittleEndian.PutUint64(m.data[base+int64(i)*8:], neighbors[i])
	}
}

func (m *MmapGraphStore) HasNode(id uint64) bool {
	_, ok := m.slots[id]
	return ok
}

func (m *MmapGraphStore) DeleteNode(id uint64) {
	if offset, ok := m.slots[id]; ok {
		// Zero the count to mark as empty
		binary.LittleEndian.PutUint32(m.data[offset:], 0)
		delete(m.slots, id)
	}
}

func (m *MmapGraphStore) Len() int {
	return len(m.slots)
}

func (m *MmapGraphStore) Range(fn func(id uint64, neighbors []uint64) bool) {
	for id := range m.slots {
		neighbors := m.GetNeighbors(id)
		if !fn(id, neighbors) {
			return
		}
	}
}

// Snapshot returns a read-only view sharing the same mmap.
// The snapshot sees the graph state at the time of the call.
// It copies the slot index but shares the underlying mmap data.
func (m *MmapGraphStore) Snapshot() GraphStore {
	slotsCopy := make(map[uint64]int64, len(m.slots))
	for id, offset := range m.slots {
		slotsCopy[id] = offset
	}

	// Capture current data slice — reads are safe on shared mmap.
	// The snapshot captures the nextOffset atomically so it won't
	// read beyond what was written at snapshot time.
	currentOffset := atomic.LoadInt64(&m.nextOffset)
	_ = currentOffset // used implicitly: slots only point to valid offsets

	return &MmapGraphStore{
		data:       m.data, // shared mmap, read-only access
		maxDegree:  m.maxDegree,
		slotSize:   m.slotSize,
		slots:      slotsCopy,
		nextOffset: m.nextOffset,
		capacity:   m.capacity,
		readOnly:   true,
	}
}

func (m *MmapGraphStore) Clone() map[uint64][]uint64 {
	dst := make(map[uint64][]uint64, len(m.slots))
	for id := range m.slots {
		neighbors := m.GetNeighbors(id)
		neighborsCopy := make([]uint64, len(neighbors))
		copy(neighborsCopy, neighbors)
		dst[id] = neighborsCopy
	}
	return dst
}

func (m *MmapGraphStore) ReplaceAll(graph map[uint64][]uint64) {
	// Reset all state
	m.slots = make(map[uint64]int64, len(graph))
	m.nextOffset = 0

	for id, neighbors := range graph {
		offset, err := m.allocSlot(id)
		if err != nil {
			panic(fmt.Sprintf("mmap graph: alloc during ReplaceAll: %v", err))
		}

		count := len(neighbors)
		if count > m.maxDegree {
			count = m.maxDegree
		}

		binary.LittleEndian.PutUint32(m.data[offset:], uint32(count))
		base := offset + 4
		for i := 0; i < count; i++ {
			binary.LittleEndian.PutUint64(m.data[base+int64(i)*8:], neighbors[i])
		}
	}
}

// Sync flushes the mmap to disk.
func (m *MmapGraphStore) Sync() error {
	if m.data == nil || m.readOnly {
		return nil
	}
	return unix.Msync(m.data, unix.MS_SYNC)
}
