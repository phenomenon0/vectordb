package index

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
)

// compactedVector holds a vector read by a parallel worker.
type compactedVector struct {
	id  uint64
	vec []float32
}

// parallelReadVectors reads all non-deleted vectors in parallel.
// Called with d.mu held, so field access is safe.
// Returns vectors sorted by ID for deterministic output.
func (d *DiskANNIndex) parallelReadVectors(ctx context.Context, allIDs []uint64) ([]compactedVector, error) {
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers < 1 {
		numWorkers = 1
	}
	if len(allIDs) < numWorkers*10 {
		numWorkers = 1 // not worth parallelizing for small sets
	}

	// Partition IDs across workers
	chunkSize := (len(allIDs) + numWorkers - 1) / numWorkers
	type workerResult struct {
		vectors []compactedVector
		err     error
	}
	results := make([]workerResult, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(allIDs) {
			end = len(allIDs)
		}
		if start >= len(allIDs) {
			break
		}

		wg.Add(1)
		go func(workerIdx int, ids []uint64) {
			defer wg.Done()

			vecs := make([]compactedVector, 0, len(ids))
			for _, id := range ids {
				select {
				case <-ctx.Done():
					results[workerIdx] = workerResult{err: ctx.Err()}
					return
				default:
				}

				if d.deleted[id] {
					continue
				}

				vec, err := d.readVectorForCompaction(id)
				if err != nil {
					results[workerIdx] = workerResult{err: fmt.Errorf("read vector %d: %w", id, err)}
					return
				}
				vecs = append(vecs, compactedVector{id: id, vec: vec})
			}
			results[workerIdx] = workerResult{vectors: vecs}
		}(w, allIDs[start:end])
	}

	wg.Wait()

	// Merge results
	total := 0
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
		total += len(r.vectors)
	}

	merged := make([]compactedVector, 0, total)
	for _, r := range results {
		merged = append(merged, r.vectors...)
	}

	// Sort by ID for deterministic output
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].id < merged[j].id
	})

	return merged, nil
}

// readVectorForCompaction reads a vector by ID from any source.
// Must be called with d.mu held.
func (d *DiskANNIndex) readVectorForCompaction(id uint64) ([]float32, error) {
	// Try memory first
	if v, ok := d.memoryVectors[id]; ok {
		out := make([]float32, len(v))
		copy(out, v)
		return out, nil
	}
	if quantized, ok := d.quantizedMemory[id]; ok {
		return d.quantizer.Dequantize(quantized)
	}
	// Disk
	return d.readFromDisk(id)
}

// writeCompactedVectors writes vectors to a temp mmap, returning the new offset index.
// This is the sequential write phase after parallel reads.
func (d *DiskANNIndex) writeCompactedVectors(vectors []compactedVector, tempData []byte, tempFile interface{ Truncate(int64) error; Fd() uintptr }) (newOffset int64, newOffsetIndex map[uint64]int64, finalTempData []byte, err error) {
	newOffsetIndex = make(map[uint64]int64, len(vectors))
	finalTempData = tempData

	for _, cv := range vectors {
		if d.quantizer != nil {
			quantized, qErr := d.quantizer.Quantize(cv.vec)
			if qErr != nil {
				return 0, nil, nil, fmt.Errorf("quantize vector %d: %w", cv.id, qErr)
			}

			recordSize := int64(8 + 4 + len(quantized))
			if newOffset+recordSize > int64(len(finalTempData)) {
				finalTempData, err = d.growTempMmap(finalTempData, tempFile, newOffset*2)
				if err != nil {
					return 0, nil, nil, err
				}
			}

			binary.LittleEndian.PutUint64(finalTempData[newOffset:], cv.id)
			binary.LittleEndian.PutUint32(finalTempData[newOffset+8:], uint32(len(quantized)))
			copy(finalTempData[newOffset+12:], quantized)
			newOffsetIndex[cv.id] = newOffset
			newOffset += recordSize
		} else {
			recordSize := int64(8 + d.dim*4)
			if newOffset+recordSize > int64(len(finalTempData)) {
				finalTempData, err = d.growTempMmap(finalTempData, tempFile, newOffset*2)
				if err != nil {
					return 0, nil, nil, err
				}
			}

			newOffsetIndex[cv.id] = newOffset
			binary.LittleEndian.PutUint64(finalTempData[newOffset:], cv.id)
			for i, val := range cv.vec {
				bits := math.Float32bits(val)
				binary.LittleEndian.PutUint32(finalTempData[newOffset+8+int64(i*4):], bits)
			}
			newOffset += recordSize
		}
	}

	return newOffset, newOffsetIndex, finalTempData, nil
}

func (d *DiskANNIndex) growTempMmap(data []byte, tempFile interface{ Truncate(int64) error; Fd() uintptr }, newSize int64) ([]byte, error) {
	if err := mmapUnmap(data); err != nil {
		return nil, fmt.Errorf("unmap for grow: %w", err)
	}
	if err := tempFile.Truncate(newSize); err != nil {
		return nil, fmt.Errorf("truncate for grow: %w", err)
	}
	newData, err := mmapCreate(int(tempFile.Fd()), int(newSize))
	if err != nil {
		return nil, fmt.Errorf("remap for grow: %w", err)
	}
	return newData, nil
}
