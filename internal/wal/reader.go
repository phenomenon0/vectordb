// Package wal provides Write-Ahead Logging for VectorDB durability.
package wal

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Reader reads entries from a WAL segment file
type Reader struct {
	r io.Reader
}

// NewReader creates a new WAL reader from an io.Reader
func NewReader(r io.Reader) *Reader {
	return &Reader{r: r}
}

// Next reads and returns the next WAL entry
func (r *Reader) Next() (*Entry, error) {
	// Read header: [4-byte length][4-byte CRC]
	header := make([]byte, 8)
	if _, err := io.ReadFull(r.r, header); err != nil {
		return nil, err
	}

	length := binary.LittleEndian.Uint32(header[0:4])
	expectedCRC := binary.LittleEndian.Uint32(header[4:8])

	// Sanity check on length (max 100MB per entry)
	if length > 100*1024*1024 {
		return nil, fmt.Errorf("entry length %d exceeds maximum", length)
	}

	// Read data
	data := make([]byte, length)
	if _, err := io.ReadFull(r.r, data); err != nil {
		return nil, err
	}

	// Verify CRC
	actualCRC := crc32.ChecksumIEEE(data)
	if actualCRC != expectedCRC {
		return nil, fmt.Errorf("CRC mismatch: expected %08x, got %08x", expectedCRC, actualCRC)
	}

	// Deserialize entry — auto-detect format for backward compatibility.
	// Old WAL entries are JSON (starts with '{'), new ones are cowrie binary.
	if isJSONPayload(data) {
		var entry Entry
		if err := json.Unmarshal(data, &entry); err != nil {
			return nil, fmt.Errorf("failed to deserialize JSON entry: %w", err)
		}
		return &entry, nil
	}
	entry, err := decodeEntryCowrie(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize cowrie entry: %w", err)
	}
	return entry, nil
}

// SegmentReader reads all entries from a specific segment file
type SegmentReader struct {
	path   string
	file   *os.File
	reader *Reader
}

// OpenSegment opens a segment file for reading
func OpenSegment(path string) (*SegmentReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	return &SegmentReader{
		path:   path,
		file:   f,
		reader: NewReader(f),
	}, nil
}

// Next reads the next entry from the segment
func (sr *SegmentReader) Next() (*Entry, error) {
	return sr.reader.Next()
}

// Close closes the segment file
func (sr *SegmentReader) Close() error {
	if sr.file != nil {
		return sr.file.Close()
	}
	return nil
}

// ReadAll reads all entries from the segment
func (sr *SegmentReader) ReadAll() ([]*Entry, error) {
	var entries []*Entry
	for {
		entry, err := sr.reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			// Return what we have so far on corruption
			return entries, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// WALIterator iterates over all entries in a WAL directory
type WALIterator struct {
	dir      string
	segments []string
	current  *SegmentReader
	index    int
}

// NewIterator creates an iterator over all WAL entries in a directory
func NewIterator(dir string) (*WALIterator, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var segments []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "wal-") && strings.HasSuffix(name, ".log") {
			segments = append(segments, filepath.Join(dir, name))
		}
	}

	// Sort by name to ensure chronological order
	sort.Strings(segments)

	return &WALIterator{
		dir:      dir,
		segments: segments,
		index:    -1,
	}, nil
}

// Next returns the next entry across all segments
func (it *WALIterator) Next() (*Entry, error) {
	for {
		// Try to read from current segment
		if it.current != nil {
			entry, err := it.current.Next()
			if err == io.EOF {
				// Move to next segment
				it.current.Close()
				it.current = nil
				continue
			}
			if err != nil {
				// Corruption, try next segment
				it.current.Close()
				it.current = nil
				continue
			}
			return entry, nil
		}

		// Open next segment
		it.index++
		if it.index >= len(it.segments) {
			return nil, io.EOF
		}

		sr, err := OpenSegment(it.segments[it.index])
		if err != nil {
			// Skip problematic segments
			continue
		}
		it.current = sr
	}
}

// Close closes the iterator
func (it *WALIterator) Close() error {
	if it.current != nil {
		return it.current.Close()
	}
	return nil
}

// ReadAll reads all entries from a WAL directory
func ReadAll(dir string) ([]*Entry, error) {
	it, err := NewIterator(dir)
	if err != nil {
		return nil, err
	}
	defer it.Close()

	var entries []*Entry
	for {
		entry, err := it.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return entries, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// Replay replays WAL entries to a handler function
// The handler returns true to continue, false to stop
func Replay(dir string, handler func(*Entry) bool) error {
	it, err := NewIterator(dir)
	if err != nil {
		return err
	}
	defer it.Close()

	for {
		entry, err := it.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if !handler(entry) {
			return nil
		}
	}
}

// ReplayFrom replays WAL entries starting from a specific LSN
func ReplayFrom(dir string, fromLSN uint64, handler func(*Entry) bool) error {
	return Replay(dir, func(entry *Entry) bool {
		if entry.LSN < fromLSN {
			return true // skip, continue
		}
		return handler(entry)
	})
}

// Stats represents statistics about a WAL directory
type Stats struct {
	SegmentCount int
	EntryCount   int64
	TotalSize    int64
	FirstLSN     uint64
	LastLSN      uint64
	FirstTime    int64
	LastTime     int64
	OpCounts     map[OpType]int64
}

// GetStats returns statistics about a WAL directory
func GetStats(dir string) (*Stats, error) {
	stats := &Stats{
		OpCounts: make(map[OpType]int64),
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "wal-") && strings.HasSuffix(name, ".log") {
			stats.SegmentCount++
			if info, err := entry.Info(); err == nil {
				stats.TotalSize += info.Size()
			}
		}
	}

	// Read all entries for detailed stats
	it, err := NewIterator(dir)
	if err != nil {
		return stats, nil
	}
	defer it.Close()

	for {
		entry, err := it.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}

		stats.EntryCount++
		stats.OpCounts[entry.Op]++

		if stats.FirstLSN == 0 || entry.LSN < stats.FirstLSN {
			stats.FirstLSN = entry.LSN
		}
		if entry.LSN > stats.LastLSN {
			stats.LastLSN = entry.LSN
		}
		if stats.FirstTime == 0 || entry.Timestamp < stats.FirstTime {
			stats.FirstTime = entry.Timestamp
		}
		if entry.Timestamp > stats.LastTime {
			stats.LastTime = entry.Timestamp
		}
	}

	return stats, nil
}
