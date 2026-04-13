//go:build windows

package index

import (
	"encoding/binary"
	"fmt"
	"os"
)

// MmapGraphStore stores the neighbor graph in a memory-mapped file.
// Each node occupies a fixed-size record: [4 bytes count][maxDegree * 8 bytes neighbor IDs].
// The ID-to-slot mapping is kept in memory (8 bytes per node).
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
	mmapGraphInitialSlots = 1 << 12 // 4K slots initial capacity (grows 2x as needed)
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

	data, err := mmapCreate(fdFromFile(file), int(initialCapacity))
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("mmap graph: mmap: %w", err)
	}

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
		if err := mmapUnmap(m.data); err != nil && firstErr == nil {
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

	if err := mmapUnmap(m.data); err != nil {
		return fmt.Errorf("mmap graph grow: unmap: %w", err)
	}

	if err := m.file.Truncate(newCap); err != nil {
		return fmt.Errorf("mmap graph grow: truncate: %w", err)
	}

	data, err := mmapCreate(fdFromFile(m.file), int(newCap))
	if err != nil {
		return fmt.Errorf("mmap graph grow: remap: %w", err)
	}

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
func (m *MmapGraphStore) Snapshot() GraphStore {
	slotsCopy := make(map[uint64]int64, len(m.slots))
	for id, offset := range m.slots {
		slotsCopy[id] = offset
	}

	return &MmapGraphStore{
		data:       m.data,
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
	m.slots = make(map[uint64]int64, len(graph))
	m.nextOffset = 0

	for id, neighbors := range graph {
		m.SetNeighbors(id, neighbors)
	}
}

// Sync flushes the mmap to disk.
func (m *MmapGraphStore) Sync() error {
	if m.data == nil || m.readOnly {
		return nil
	}
	return mmapSync(m.data)
}
