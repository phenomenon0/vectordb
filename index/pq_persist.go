package index

import (
	"encoding/binary"
	"fmt"
	"math"
)

// PQ persistence format:
//   [4] magic "PQ01"
//   [4] dim
//   [4] m (num subvectors)
//   [4] ksub (centroids per subvector)
//   [4] dsub (dimension per subvector)
//   [1] trained (0 or 1)
//   [4] num_vectors
//   [4] codes_len
//   codebooks: m * ksub * dsub * 4 bytes (float32)
//   codes: codes_len bytes
//   ids: num_vectors * 8 bytes (uint64)

var pqMagic = [4]byte{'P', 'Q', '0', '1'}

func exportPQData(dim, m, ksub, dsub int, trained bool, codebooks [][][]float32, codes []byte, ids []uint64) ([]byte, error) {
	// Calculate total size
	codebookBytes := m * ksub * dsub * 4
	headerBytes := 4 + 4 + 4 + 4 + 4 + 1 + 4 + 4 // magic + dim + m + ksub + dsub + trained + numVecs + codesLen
	totalSize := headerBytes + codebookBytes + len(codes) + len(ids)*8

	buf := make([]byte, 0, totalSize)

	// Magic
	buf = append(buf, pqMagic[:]...)

	// Header
	tmp := make([]byte, 8)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(dim))
	buf = append(buf, tmp[:4]...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(m))
	buf = append(buf, tmp[:4]...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(ksub))
	buf = append(buf, tmp[:4]...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(dsub))
	buf = append(buf, tmp[:4]...)

	if trained {
		buf = append(buf, 1)
	} else {
		buf = append(buf, 0)
	}

	binary.LittleEndian.PutUint32(tmp[:4], uint32(len(ids)))
	buf = append(buf, tmp[:4]...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(len(codes)))
	buf = append(buf, tmp[:4]...)

	// Codebooks
	for i := 0; i < m; i++ {
		for k := 0; k < ksub; k++ {
			if i < len(codebooks) && k < len(codebooks[i]) {
				for d := 0; d < dsub; d++ {
					if d < len(codebooks[i][k]) {
						binary.LittleEndian.PutUint32(tmp[:4], math.Float32bits(codebooks[i][k][d]))
					} else {
						binary.LittleEndian.PutUint32(tmp[:4], 0)
					}
					buf = append(buf, tmp[:4]...)
				}
			} else {
				// Zero-pad missing codebook entries
				for d := 0; d < dsub; d++ {
					buf = append(buf, 0, 0, 0, 0)
				}
			}
		}
	}

	// Codes
	buf = append(buf, codes...)

	// IDs
	for _, id := range ids {
		binary.LittleEndian.PutUint64(tmp[:8], id)
		buf = append(buf, tmp[:8]...)
	}

	return buf, nil
}

func importPQData(data []byte) (dim, m, ksub, dsub int, trained bool, codebooks [][][]float32, codes []byte, ids []uint64, err error) {
	minHeader := 4 + 4 + 4 + 4 + 4 + 1 + 4 + 4 // 29 bytes
	if len(data) < minHeader {
		err = fmt.Errorf("data too short: %d bytes", len(data))
		return
	}

	off := 0

	// Magic
	if data[0] != pqMagic[0] || data[1] != pqMagic[1] || data[2] != pqMagic[2] || data[3] != pqMagic[3] {
		err = fmt.Errorf("invalid magic: expected PQ01")
		return
	}
	off += 4

	dim = int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	m = int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	ksub = int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	dsub = int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	trained = data[off] == 1
	off++

	numVecs := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	codesLen := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	// Validate sizes
	codebookBytes := m * ksub * dsub * 4
	expectedSize := off + codebookBytes + codesLen + numVecs*8
	if len(data) < expectedSize {
		err = fmt.Errorf("data too short: have %d, need %d", len(data), expectedSize)
		return
	}

	// Read codebooks
	codebooks = make([][][]float32, m)
	for i := 0; i < m; i++ {
		codebooks[i] = make([][]float32, ksub)
		for k := 0; k < ksub; k++ {
			codebooks[i][k] = make([]float32, dsub)
			for d := 0; d < dsub; d++ {
				codebooks[i][k][d] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
				off += 4
			}
		}
	}

	// Read codes
	codes = make([]byte, codesLen)
	copy(codes, data[off:off+codesLen])
	off += codesLen

	// Read IDs
	ids = make([]uint64, numVecs)
	for i := 0; i < numVecs; i++ {
		ids[i] = binary.LittleEndian.Uint64(data[off:])
		off += 8
	}

	return
}
