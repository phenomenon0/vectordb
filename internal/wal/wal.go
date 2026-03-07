// Package wal provides Write-Ahead Logging for VectorDB durability.
//
// The WAL ensures that all write operations are persisted to disk before
// being acknowledged, enabling crash recovery without data loss.
//
// WAL Format:
//   - Each entry is length-prefixed with a CRC32 checksum
//   - Format: [4-byte length][4-byte CRC32][entry data]
//   - Entries are append-only within a segment file
//   - Segment files are rotated when they exceed MaxSegmentSize
package wal

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Operation types for WAL entries
type OpType uint8

const (
	OpInsert OpType = iota + 1
	OpDelete
	OpUpdate
	OpBatchInsert
	OpCreateCollection
	OpDeleteCollection
	OpCompact
)

func (op OpType) String() string {
	switch op {
	case OpInsert:
		return "INSERT"
	case OpDelete:
		return "DELETE"
	case OpUpdate:
		return "UPDATE"
	case OpBatchInsert:
		return "BATCH_INSERT"
	case OpCreateCollection:
		return "CREATE_COLLECTION"
	case OpDeleteCollection:
		return "DELETE_COLLECTION"
	case OpCompact:
		return "COMPACT"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", op)
	}
}

// Entry represents a single WAL entry
type Entry struct {
	// Sequence number (monotonically increasing)
	LSN uint64 `json:"lsn"`

	// Operation type
	Op OpType `json:"op"`

	// Timestamp when entry was created
	Timestamp int64 `json:"ts"`

	// Collection name (if applicable)
	Collection string `json:"collection,omitempty"`

	// Document ID (for insert/delete/update)
	ID string `json:"id,omitempty"`

	// Vector data (for insert/update)
	Vector []float32 `json:"vector,omitempty"`

	// Document content (for insert/update)
	Doc string `json:"doc,omitempty"`

	// Metadata (for insert/update)
	Meta map[string]string `json:"meta,omitempty"`

	// Tenant ID (for multi-tenancy)
	TenantID string `json:"tenant_id,omitempty"`

	// Batch data (for batch insert)
	Batch []BatchEntry `json:"batch,omitempty"`
}

// BatchEntry represents a single item in a batch insert
type BatchEntry struct {
	ID     string            `json:"id"`
	Vector []float32         `json:"vector"`
	Doc    string            `json:"doc,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

// Config holds WAL configuration
type Config struct {
	// Directory where WAL files are stored
	Dir string

	// Maximum size of a single WAL segment file (default: 64MB)
	MaxSegmentSize int64

	// Whether to fsync after each write (default: true)
	SyncOnWrite bool

	// Interval for periodic sync if SyncOnWrite is false
	SyncInterval time.Duration

	// Maximum number of segments to keep (0 = unlimited)
	MaxSegments int

	// Callback when a segment is rotated
	OnRotate func(oldSegment string)
}

// DefaultConfig returns sensible defaults
func DefaultConfig(dir string) Config {
	return Config{
		Dir:            dir,
		MaxSegmentSize: 64 * 1024 * 1024, // 64MB
		SyncOnWrite:    true,
		SyncInterval:   time.Second,
		MaxSegments:    0, // unlimited
	}
}

// WAL is the Write-Ahead Log
type WAL struct {
	mu     sync.Mutex
	config Config

	// Current segment file
	segment        *os.File
	currentSegPath string
	segmentSize    int64
	segmentNum     uint64

	// Current LSN (Log Sequence Number)
	lsn uint64

	// Closed flag
	closed bool

	// Background sync goroutine
	syncDone chan struct{}
}

// Open opens or creates a WAL in the specified directory
func Open(config Config) (*WAL, error) {
	if config.Dir == "" {
		return nil, errors.New("WAL directory is required")
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(config.Dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	// Apply defaults
	if config.MaxSegmentSize <= 0 {
		config.MaxSegmentSize = 64 * 1024 * 1024
	}

	w := &WAL{
		config:   config,
		syncDone: make(chan struct{}),
	}

	// Find the latest segment and LSN
	if err := w.recover(); err != nil {
		return nil, fmt.Errorf("WAL recovery failed: %w", err)
	}

	// Open/create current segment
	if err := w.openSegment(); err != nil {
		return nil, fmt.Errorf("failed to open WAL segment: %w", err)
	}

	// Start background sync if not syncing on every write
	if !config.SyncOnWrite && config.SyncInterval > 0 {
		go w.backgroundSync()
	}

	return w, nil
}

// recover scans existing WAL files to find the latest LSN
func (w *WAL) recover() error {
	entries, err := os.ReadDir(w.config.Dir)
	if err != nil {
		return err
	}

	var segments []uint64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "wal-") && strings.HasSuffix(name, ".log") {
			numStr := strings.TrimPrefix(strings.TrimSuffix(name, ".log"), "wal-")
			if num, err := strconv.ParseUint(numStr, 10, 64); err == nil {
				segments = append(segments, num)
			}
		}
	}

	if len(segments) == 0 {
		// No existing segments, start fresh
		w.segmentNum = 1
		w.lsn = 0
		return nil
	}

	// Sort segments to find the latest
	sort.Slice(segments, func(i, j int) bool { return segments[i] < segments[j] })
	w.segmentNum = segments[len(segments)-1]

	// Scan the latest segment to find the last LSN
	latestPath := w.segmentPath(w.segmentNum)
	lastLSN, err := w.scanSegmentForLastLSN(latestPath)
	if err != nil {
		return fmt.Errorf("failed to scan latest segment: %w", err)
	}
	w.lsn = lastLSN

	return nil
}

// scanSegmentForLastLSN reads a segment file and returns the last LSN
func (w *WAL) scanSegmentForLastLSN(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()

	var lastLSN uint64
	reader := NewReader(f)

	for {
		entry, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			// Corrupted entry, stop here
			break
		}
		lastLSN = entry.LSN
	}

	return lastLSN, nil
}

// segmentPath returns the path for a segment number
func (w *WAL) segmentPath(num uint64) string {
	return filepath.Join(w.config.Dir, fmt.Sprintf("wal-%08d.log", num))
}

// openSegment opens the current segment file for writing
func (w *WAL) openSegment() error {
	path := w.segmentPath(w.segmentNum)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}

	// Get current size
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}

	w.segment = f
	w.currentSegPath = path
	w.segmentSize = info.Size()
	return nil
}

// Write appends an entry to the WAL
func (w *WAL) Write(entry *Entry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("WAL is closed")
	}

	// Assign LSN and timestamp
	w.lsn++
	entry.LSN = w.lsn
	if entry.Timestamp == 0 {
		entry.Timestamp = time.Now().UnixNano()
	}

	// Serialize entry
	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to serialize WAL entry: %w", err)
	}

	// Write with length prefix and CRC
	if err := w.writeEntry(data); err != nil {
		return err
	}

	// Check if we need to rotate
	if w.segmentSize >= w.config.MaxSegmentSize {
		if err := w.rotate(); err != nil {
			return fmt.Errorf("failed to rotate WAL: %w", err)
		}
	}

	return nil
}

// writeEntry writes a single entry with length prefix and CRC
func (w *WAL) writeEntry(data []byte) error {
	// Calculate CRC
	crc := crc32.ChecksumIEEE(data)

	// Write header: [4-byte length][4-byte CRC]
	header := make([]byte, 8)
	binary.LittleEndian.PutUint32(header[0:4], uint32(len(data)))
	binary.LittleEndian.PutUint32(header[4:8], crc)

	// Write header
	if _, err := w.segment.Write(header); err != nil {
		return err
	}

	// Write data
	if _, err := w.segment.Write(data); err != nil {
		return err
	}

	w.segmentSize += int64(len(header) + len(data))

	// Sync if configured
	if w.config.SyncOnWrite {
		if err := w.segment.Sync(); err != nil {
			return fmt.Errorf("fsync failed: %w", err)
		}
	}

	return nil
}

// rotate closes the current segment and opens a new one
func (w *WAL) rotate() error {
	oldPath := w.currentSegPath

	// Close current segment
	if err := w.segment.Sync(); err != nil {
		return err
	}
	if err := w.segment.Close(); err != nil {
		return err
	}

	// Increment segment number
	w.segmentNum++

	// Open new segment
	if err := w.openSegment(); err != nil {
		return err
	}

	// Callback
	if w.config.OnRotate != nil {
		w.config.OnRotate(oldPath)
	}

	// Clean up old segments if configured
	if w.config.MaxSegments > 0 {
		w.cleanOldSegments()
	}

	return nil
}

// cleanOldSegments removes segments exceeding MaxSegments
func (w *WAL) cleanOldSegments() {
	entries, err := os.ReadDir(w.config.Dir)
	if err != nil {
		return
	}

	var segments []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "wal-") {
			segments = append(segments, filepath.Join(w.config.Dir, entry.Name()))
		}
	}

	if len(segments) <= w.config.MaxSegments {
		return
	}

	// Sort by name (which includes sequence number)
	sort.Strings(segments)

	// Remove oldest segments
	toRemove := len(segments) - w.config.MaxSegments
	for i := 0; i < toRemove; i++ {
		os.Remove(segments[i])
	}
}

// backgroundSync periodically syncs the WAL to disk
func (w *WAL) backgroundSync() {
	ticker := time.NewTicker(w.config.SyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			w.mu.Lock()
			if w.segment != nil && !w.closed {
				w.segment.Sync()
			}
			w.mu.Unlock()
		case <-w.syncDone:
			return
		}
	}
}

// Sync forces a sync of the current segment to disk
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("WAL is closed")
	}

	return w.segment.Sync()
}

// Close closes the WAL
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}

	w.closed = true
	close(w.syncDone)

	if w.segment != nil {
		w.segment.Sync()
		return w.segment.Close()
	}

	return nil
}

// LSN returns the current Log Sequence Number
func (w *WAL) LSN() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lsn
}

// SegmentCount returns the number of WAL segment files
func (w *WAL) SegmentCount() int {
	entries, err := os.ReadDir(w.config.Dir)
	if err != nil {
		return 0
	}

	count := 0
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "wal-") {
			count++
		}
	}
	return count
}

// Size returns the total size of all WAL files
func (w *WAL) Size() int64 {
	entries, err := os.ReadDir(w.config.Dir)
	if err != nil {
		return 0
	}

	var total int64
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "wal-") {
			if info, err := entry.Info(); err == nil {
				total += info.Size()
			}
		}
	}
	return total
}

// Truncate removes all WAL segments up to (but not including) the given LSN
func (w *WAL) Truncate(beforeLSN uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	entries, err := os.ReadDir(w.config.Dir)
	if err != nil {
		return err
	}

	var segments []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "wal-") {
			segments = append(segments, filepath.Join(w.config.Dir, entry.Name()))
		}
	}

	sort.Strings(segments)

	// Check each segment's max LSN and remove if all entries are before beforeLSN
	for _, segPath := range segments {
		// Don't remove current segment
		if segPath == w.currentSegPath {
			continue
		}

		maxLSN, err := w.scanSegmentForLastLSN(segPath)
		if err != nil {
			continue
		}

		if maxLSN < beforeLSN {
			os.Remove(segPath)
		}
	}

	return nil
}
