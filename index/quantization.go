package index

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// QuantizationType represents different quantization methods
type QuantizationType int

const (
	QuantizationNone QuantizationType = iota
	QuantizationFloat16                     // 16-bit floating point (50% memory)
	QuantizationUint8                       // 8-bit unsigned integer (75% memory savings)
	QuantizationProduct                     // Product Quantization (90%+ memory savings)
	QuantizationBinary                      // Binary Quantization (97% memory savings, 32x compression)
	QuantizationBinaryMean                  // Binary with learned thresholds (better recall)
)

// Quantizer handles vector quantization and dequantization
type Quantizer interface {
	// Quantize converts float32 vectors to quantized format
	Quantize(vectors []float32) ([]byte, error)

	// Dequantize converts quantized bytes back to float32
	Dequantize(data []byte) ([]float32, error)

	// Type returns the quantization type
	Type() QuantizationType

	// BytesPerVector returns storage size per vector
	BytesPerVector() int
}

// ======================================================================================
// Float16 Quantization (50% memory savings)
// ======================================================================================

// Float16Quantizer quantizes to 16-bit floats
type Float16Quantizer struct {
	dim int
}

// NewFloat16Quantizer creates a new float16 quantizer
func NewFloat16Quantizer(dim int) *Float16Quantizer {
	return &Float16Quantizer{dim: dim}
}

func (q *Float16Quantizer) Type() QuantizationType {
	return QuantizationFloat16
}

func (q *Float16Quantizer) BytesPerVector() int {
	return q.dim * 2 // 2 bytes per dimension
}

func (q *Float16Quantizer) Quantize(vectors []float32) ([]byte, error) {
	if len(vectors)%q.dim != 0 {
		return nil, fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / q.dim
	data := make([]byte, numVectors*q.BytesPerVector())

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.dim; j++ {
			val := vectors[i*q.dim+j]
			f16 := float32ToFloat16(val)

			offset := i*q.BytesPerVector() + j*2
			data[offset] = byte(f16)
			data[offset+1] = byte(f16 >> 8)
		}
	}

	return data, nil
}

func (q *Float16Quantizer) Dequantize(data []byte) ([]float32, error) {
	if len(data)%q.BytesPerVector() != 0 {
		return nil, fmt.Errorf("invalid data length for float16 quantization")
	}

	numVectors := len(data) / q.BytesPerVector()
	vectors := make([]float32, numVectors*q.dim)

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.dim; j++ {
			offset := i*q.BytesPerVector() + j*2
			f16 := uint16(data[offset]) | uint16(data[offset+1])<<8

			vectors[i*q.dim+j] = float16ToFloat32(f16)
		}
	}

	return vectors, nil
}

// float32ToFloat16 converts float32 to IEEE 754 float16
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)

	sign := (bits >> 31) & 0x1
	exp32 := int32((bits >> 23) & 0xFF)
	frac := (bits >> 13) & 0x3FF // 10-bit mantissa

	// Rebias exponent: float32 bias=127, float16 bias=15
	exp := exp32 - 127 + 15

	// Handle special cases
	if exp <= 0 {
		// Subnormal or zero
		return uint16(sign << 15)
	} else if exp >= 31 {
		// Overflow to infinity (exponent too large for float16)
		return uint16((sign << 15) | 0x7C00)
	}

	// Normal number
	return uint16((sign << 15) | (uint32(exp) << 10) | frac)
}

// float16ToFloat32 converts IEEE 754 float16 to float32
func float16ToFloat32(f16 uint16) float32 {
	sign := uint32((f16 >> 15) & 0x1)
	exp := uint32((f16 >> 10) & 0x1F)
	frac := uint32(f16 & 0x3FF)

	// Handle special cases
	if exp == 0 {
		// Zero or subnormal
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: flush to zero for simplicity
		return 0.0
	} else if exp == 31 {
		// Infinity or NaN
		if frac == 0 {
			// Infinity
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		// NaN
		return math.Float32frombits((sign << 31) | 0x7F800000 | (frac << 13))
	}

	// Normal number
	exp32 := exp - 15 + 127 // Rebias exponent to float32
	bits := (sign << 31) | (exp32 << 23) | (frac << 13)

	return math.Float32frombits(bits)
}

// ======================================================================================
// Uint8 Scalar Quantization (75% memory savings)
// ======================================================================================

// Uint8Quantizer quantizes to 8-bit unsigned integers using min-max scaling
type Uint8Quantizer struct {
	dim    int
	min    []float32 // Min value per dimension (for dequantization)
	max    []float32 // Max value per dimension (for dequantization)
	trained bool
}

// NewUint8Quantizer creates a new uint8 quantizer
func NewUint8Quantizer(dim int) *Uint8Quantizer {
	return &Uint8Quantizer{
		dim:    dim,
		min:    make([]float32, dim),
		max:    make([]float32, dim),
		trained: false,
	}
}

func (q *Uint8Quantizer) Type() QuantizationType {
	return QuantizationUint8
}

func (q *Uint8Quantizer) BytesPerVector() int {
	return q.dim // 1 byte per dimension
}

// Train computes min/max values for each dimension
func (q *Uint8Quantizer) Train(vectors []float32) error {
	if len(vectors)%q.dim != 0 {
		return fmt.Errorf("vector length must be multiple of dimension")
	}

	if len(vectors) == 0 {
		return fmt.Errorf("need at least one vector to train")
	}

	numVectors := len(vectors) / q.dim

	// Initialize min/max with first vector
	for j := 0; j < q.dim; j++ {
		q.min[j] = vectors[j]
		q.max[j] = vectors[j]
	}

	// Find min/max across all vectors
	for i := 1; i < numVectors; i++ {
		for j := 0; j < q.dim; j++ {
			val := vectors[i*q.dim+j]
			if val < q.min[j] {
				q.min[j] = val
			}
			if val > q.max[j] {
				q.max[j] = val
			}
		}
	}

	q.trained = true
	return nil
}

func (q *Uint8Quantizer) Quantize(vectors []float32) ([]byte, error) {
	if !q.trained {
		return nil, fmt.Errorf("quantizer must be trained before quantization")
	}

	if len(vectors)%q.dim != 0 {
		return nil, fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / q.dim
	data := make([]byte, numVectors*q.BytesPerVector())

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.dim; j++ {
			val := vectors[i*q.dim+j]

			// Scale to [0, 255]
			scale := q.max[j] - q.min[j]
			if scale == 0 {
				data[i*q.dim+j] = 0
			} else {
				normalized := (val - q.min[j]) / scale
				quantized := uint8(normalized * 255.0)
				data[i*q.dim+j] = quantized
			}
		}
	}

	return data, nil
}

func (q *Uint8Quantizer) Dequantize(data []byte) ([]float32, error) {
	if !q.trained {
		return nil, fmt.Errorf("quantizer must be trained before dequantization")
	}

	if len(data)%q.BytesPerVector() != 0 {
		return nil, fmt.Errorf("invalid data length for uint8 quantization")
	}

	numVectors := len(data) / q.BytesPerVector()
	vectors := make([]float32, numVectors*q.dim)

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.dim; j++ {
			quantized := data[i*q.dim+j]

			// Descale from [0, 255] to original range
			scale := q.max[j] - q.min[j]
			normalized := float32(quantized) / 255.0
			val := normalized*scale + q.min[j]

			vectors[i*q.dim+j] = val
		}
	}

	return vectors, nil
}

// GetCodebook returns the min/max codebook for persistence
func (q *Uint8Quantizer) GetCodebook() (min []float32, max []float32) {
	return q.min, q.max
}

// SetCodebook sets the min/max codebook (for loading persisted quantizer)
func (q *Uint8Quantizer) SetCodebook(min []float32, max []float32) error {
	if len(min) != q.dim || len(max) != q.dim {
		return fmt.Errorf("codebook dimension mismatch")
	}

	q.min = min
	q.max = max
	q.trained = true

	return nil
}

// ======================================================================================
// Product Quantization (90%+ memory savings)
// ======================================================================================

// ProductQuantizer uses k-means clustering to quantize subvectors
type ProductQuantizer struct {
	dim         int     // Vector dimension
	m           int     // Number of subvectors
	ksub        int     // Codebook size per subvector (default: 256 for 8-bit codes)
	dsub        int     // Dimension per subvector (dim / m)
	codebooks   [][][]float32 // [m][ksub][dsub] - m codebooks, each with ksub centroids of dsub dimensions
	trained     bool
}

// NewProductQuantizer creates a new product quantizer
// m: number of subvectors (typically 8, 16, 32)
// ksub: codebook size (typically 256 for 8-bit codes)
func NewProductQuantizer(dim int, m int, ksub int) (*ProductQuantizer, error) {
	if dim%m != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by m=%d", dim, m)
	}

	return &ProductQuantizer{
		dim:       dim,
		m:         m,
		ksub:      ksub,
		dsub:      dim / m,
		codebooks: make([][][]float32, m),
		trained:   false,
	}, nil
}

func (q *ProductQuantizer) Type() QuantizationType {
	return QuantizationProduct
}

func (q *ProductQuantizer) BytesPerVector() int {
	// Each subvector uses 1 byte (for ksub=256)
	return q.m
}

// Train trains the product quantizer using k-means clustering
func (q *ProductQuantizer) Train(vectors []float32, iterations int) error {
	if len(vectors)%q.dim != 0 {
		return fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / q.dim
	if numVectors < q.ksub {
		return fmt.Errorf("need at least %d vectors to train, got %d", q.ksub, numVectors)
	}

	// Train a codebook for each subvector
	for i := 0; i < q.m; i++ {
		// Extract subvectors for this partition
		subvectors := make([][]float32, numVectors)
		for j := 0; j < numVectors; j++ {
			subvec := make([]float32, q.dsub)
			copy(subvec, vectors[j*q.dim+i*q.dsub:(j*q.dim+(i+1)*q.dsub)])
			subvectors[j] = subvec
		}

		// Run k-means to get codebook
		flatSubvecs := make([]float32, numVectors*q.dsub)
		for j, sv := range subvectors {
			copy(flatSubvecs[j*q.dsub:], sv)
		}

		centroids, err := kMeansSimple(flatSubvecs, q.ksub, q.dsub, iterations)
		if err != nil {
			return fmt.Errorf("failed to train subvector %d: %w", i, err)
		}

		q.codebooks[i] = centroids
	}

	q.trained = true
	return nil
}

func (q *ProductQuantizer) Quantize(vectors []float32) ([]byte, error) {
	if !q.trained {
		return nil, fmt.Errorf("quantizer must be trained before quantization")
	}

	if len(vectors)%q.dim != 0 {
		return nil, fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / q.dim
	codes := make([]byte, numVectors*q.m)

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.m; j++ {
			// Extract subvector
			subvec := vectors[i*q.dim+j*q.dsub : i*q.dim+(j+1)*q.dsub]

			// Find nearest centroid in codebook j
			minDist := float32(math.Inf(1))
			minIdx := 0

			for k := 0; k < q.ksub; k++ {
				dist := euclideanDistanceSimple(subvec, q.codebooks[j][k])
				if dist < minDist {
					minDist = dist
					minIdx = k
				}
			}

			codes[i*q.m+j] = byte(minIdx)
		}
	}

	return codes, nil
}

func (q *ProductQuantizer) Dequantize(data []byte) ([]float32, error) {
	if !q.trained {
		return nil, fmt.Errorf("quantizer must be trained before dequantization")
	}

	if len(data)%q.m != 0 {
		return nil, fmt.Errorf("invalid data length for product quantization")
	}

	numVectors := len(data) / q.m
	vectors := make([]float32, numVectors*q.dim)

	for i := 0; i < numVectors; i++ {
		for j := 0; j < q.m; j++ {
			code := int(data[i*q.m+j])
			centroid := q.codebooks[j][code]

			// Copy centroid to output vector
			copy(vectors[i*q.dim+j*q.dsub:], centroid)
		}
	}

	return vectors, nil
}

// GetCodebooks returns the trained codebooks for persistence
func (q *ProductQuantizer) GetCodebooks() [][][]float32 {
	return q.codebooks
}

// SetCodebooks sets the codebooks (for loading persisted quantizer)
func (q *ProductQuantizer) SetCodebooks(codebooks [][][]float32) error {
	if len(codebooks) != q.m {
		return fmt.Errorf("codebooks count mismatch: expected %d, got %d", q.m, len(codebooks))
	}

	q.codebooks = codebooks
	q.trained = true

	return nil
}

// ======================================================================================
// Helper Functions
// ======================================================================================

// kMeansSimple performs simple k-means clustering
func kMeansSimple(vectors []float32, k int, dim int, maxIter int) ([][]float32, error) {
	if len(vectors)%dim != 0 {
		return nil, fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / dim
	if numVectors < k {
		return nil, fmt.Errorf("not enough vectors for k-means: need %d, have %d", k, numVectors)
	}

	// Initialize centroids randomly
	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		idx := i * numVectors / k // Evenly spaced initialization
		copy(centroids[i], vectors[idx*dim:(idx+1)*dim])
	}

	// Iterate
	for iter := 0; iter < maxIter; iter++ {
		// Assignment step
		assignments := make([]int, numVectors)
		for i := 0; i < numVectors; i++ {
			vec := vectors[i*dim : (i+1)*dim]
			minDist := float32(math.Inf(1))
			minIdx := 0

			for j := 0; j < k; j++ {
				dist := euclideanDistanceSimple(vec, centroids[j])
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}

			assignments[i] = minIdx
		}

		// Update step
		newCentroids := make([][]float32, k)
		counts := make([]int, k)

		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float32, dim)
		}

		for i := 0; i < numVectors; i++ {
			clusterIdx := assignments[i]
			vec := vectors[i*dim : (i+1)*dim]

			for d := 0; d < dim; d++ {
				newCentroids[clusterIdx][d] += vec[d]
			}
			counts[clusterIdx]++
		}

		// Average
		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for d := 0; d < dim; d++ {
					newCentroids[i][d] /= float32(counts[i])
				}
			}
		}

		centroids = newCentroids
	}

	return centroids, nil
}

func euclideanDistanceSimple(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// QuantizationInfo contains metadata about quantization
type QuantizationInfo struct {
	Type           QuantizationType
	OriginalSize   int // Bytes before quantization
	CompressedSize int // Bytes after quantization
	CompressionRatio float64
	Dimension      int
	VectorCount    int
}

// GetQuantizationInfo calculates compression statistics
func GetQuantizationInfo(quantizer Quantizer, numVectors int) QuantizationInfo {
	dim := 0
	if q, ok := quantizer.(*Float16Quantizer); ok {
		dim = q.dim
	} else if q, ok := quantizer.(*Uint8Quantizer); ok {
		dim = q.dim
	} else if q, ok := quantizer.(*ProductQuantizer); ok {
		dim = q.dim
	}

	originalSize := numVectors * dim * 4 // float32 = 4 bytes
	compressedSize := numVectors * quantizer.BytesPerVector()

	ratio := float64(originalSize) / float64(compressedSize)

	return QuantizationInfo{
		Type:           quantizer.Type(),
		OriginalSize:   originalSize,
		CompressedSize: compressedSize,
		CompressionRatio: ratio,
		Dimension:      dim,
		VectorCount:    numVectors,
	}
}

// ======================================================================================
// Binary Quantization (97% memory savings, 32x compression)
// ======================================================================================
// Converts each float32 dimension to a single bit (1 if > threshold, 0 otherwise)
// Uses Hamming distance for similarity (popcount - extremely fast)
// Best for: High-dimensional vectors (768d+), reranking candidates
// ======================================================================================

// BinaryQuantizer converts vectors to binary codes (1 bit per dimension)
type BinaryQuantizer struct {
	dim        int
	thresholds []float32 // Per-dimension thresholds (default: 0 for normalized vectors)
	trained    bool
}

// NewBinaryQuantizer creates a binary quantizer with zero threshold (for normalized vectors)
func NewBinaryQuantizer(dim int) *BinaryQuantizer {
	return &BinaryQuantizer{
		dim:        dim,
		thresholds: make([]float32, dim), // Zero-initialized = sign quantization
		trained:    true,                 // Works without training for normalized vectors
	}
}

func (q *BinaryQuantizer) Type() QuantizationType {
	return QuantizationBinary
}

func (q *BinaryQuantizer) BytesPerVector() int {
	return (q.dim + 7) / 8 // Ceiling division: bits to bytes
}

// Train learns optimal thresholds from data (per-dimension median)
// This improves recall by ~5-10% over zero threshold
func (q *BinaryQuantizer) Train(vectors []float32) error {
	if len(vectors)%q.dim != 0 {
		return fmt.Errorf("vector length must be multiple of dimension")
	}
	
	numVectors := len(vectors) / q.dim
	if numVectors < 10 {
		return fmt.Errorf("need at least 10 vectors to train, got %d", numVectors)
	}

	// Compute per-dimension mean (approximates median for normal distributions)
	for d := 0; d < q.dim; d++ {
		var sum float32
		for i := 0; i < numVectors; i++ {
			sum += vectors[i*q.dim+d]
		}
		q.thresholds[d] = sum / float32(numVectors)
	}
	
	q.trained = true
	return nil
}

func (q *BinaryQuantizer) Quantize(vectors []float32) ([]byte, error) {
	if len(vectors)%q.dim != 0 {
		return nil, fmt.Errorf("vector length must be multiple of dimension")
	}

	numVectors := len(vectors) / q.dim
	bytesPerVec := q.BytesPerVector()
	data := make([]byte, numVectors*bytesPerVec)

	for i := 0; i < numVectors; i++ {
		vecStart := i * q.dim
		outStart := i * bytesPerVec
		
		for d := 0; d < q.dim; d++ {
			if vectors[vecStart+d] > q.thresholds[d] {
				// Set bit d
				byteIdx := d / 8
				bitIdx := uint(d % 8)
				data[outStart+byteIdx] |= 1 << bitIdx
			}
		}
	}

	return data, nil
}

func (q *BinaryQuantizer) Dequantize(data []byte) ([]float32, error) {
	bytesPerVec := q.BytesPerVector()
	if len(data)%bytesPerVec != 0 {
		return nil, fmt.Errorf("invalid data length for binary quantization")
	}

	numVectors := len(data) / bytesPerVec
	vectors := make([]float32, numVectors*q.dim)

	for i := 0; i < numVectors; i++ {
		inStart := i * bytesPerVec
		outStart := i * q.dim
		
		for d := 0; d < q.dim; d++ {
			byteIdx := d / 8
			bitIdx := uint(d % 8)
			if (data[inStart+byteIdx]>>bitIdx)&1 == 1 {
				vectors[outStart+d] = 1.0
			} else {
				vectors[outStart+d] = -1.0
			}
		}
	}

	return vectors, nil
}

// GetThresholds returns the learned thresholds for persistence
func (q *BinaryQuantizer) GetThresholds() []float32 {
	return q.thresholds
}

// SetThresholds sets thresholds (for loading persisted quantizer)
func (q *BinaryQuantizer) SetThresholds(thresholds []float32) error {
	if len(thresholds) != q.dim {
		return fmt.Errorf("threshold dimension mismatch: expected %d, got %d", q.dim, len(thresholds))
	}
	q.thresholds = thresholds
	q.trained = true
	return nil
}

// ======================================================================================
// Binary Distance Functions (Hamming distance using popcount)
// ======================================================================================

// HammingDistance computes Hamming distance between two binary vectors
// Returns number of differing bits
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		return -1
	}
	
	dist := 0
	for i := range a {
		dist += popcount(a[i] ^ b[i])
	}
	return dist
}

// HammingDistanceBatch computes Hamming distances from query to multiple vectors
// Returns distances in same order as vectors
// Optimized with 8-byte (uint64) processing for better performance
func HammingDistanceBatch(query []byte, vectors []byte, bytesPerVec int) []int {
	if len(vectors)%bytesPerVec != 0 {
		return nil
	}
	
	numVectors := len(vectors) / bytesPerVec
	distances := make([]int, numVectors)
	
	// Process 8 bytes at a time using uint64
	chunks := bytesPerVec / 8
	_ = bytesPerVec % 8 // remainder handled in loop
	
	// Pre-convert query to uint64 chunks
	queryChunks := make([]uint64, chunks)
	for c := 0; c < chunks; c++ {
		offset := c * 8
		queryChunks[c] = uint64(query[offset]) |
			uint64(query[offset+1])<<8 |
			uint64(query[offset+2])<<16 |
			uint64(query[offset+3])<<24 |
			uint64(query[offset+4])<<32 |
			uint64(query[offset+5])<<40 |
			uint64(query[offset+6])<<48 |
			uint64(query[offset+7])<<56
	}
	
	for i := 0; i < numVectors; i++ {
		start := i * bytesPerVec
		dist := 0
		
		// Process 8-byte chunks
		for c := 0; c < chunks; c++ {
			offset := start + c*8
			vecChunk := uint64(vectors[offset]) |
				uint64(vectors[offset+1])<<8 |
				uint64(vectors[offset+2])<<16 |
				uint64(vectors[offset+3])<<24 |
				uint64(vectors[offset+4])<<32 |
				uint64(vectors[offset+5])<<40 |
				uint64(vectors[offset+6])<<48 |
				uint64(vectors[offset+7])<<56
			
			dist += popcount64(queryChunks[c] ^ vecChunk)
		}
		
		// Handle remaining bytes
		for j := chunks * 8; j < bytesPerVec; j++ {
			dist += popcount(query[j] ^ vectors[start+j])
		}
		
		distances[i] = dist
	}
	
	return distances
}

// popcount64 counts set bits in uint64 using parallel bit counting
func popcount64(x uint64) int {
	// Use Brian Kernighan's algorithm for sparse bits
	// Or parallel counting for dense bits
	x = x - ((x >> 1) & 0x5555555555555555)
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
	x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
	return int((x * 0x0101010101010101) >> 56)
}

// HammingToCosineSimilarity converts Hamming distance to approximate cosine similarity
// For binary codes from normalized vectors: cos(theta) ≈ 1 - 2*hamming/dim
func HammingToCosineSimilarity(hammingDist int, dim int) float32 {
	return 1.0 - 2.0*float32(hammingDist)/float32(dim)
}

// popcount counts set bits in a byte (Hamming weight)
// Uses lookup table for speed
func popcount(b byte) int {
	return int(popcountTable[b])
}

// Precomputed popcount lookup table
var popcountTable = [256]byte{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
}

// ======================================================================================
// Rescoring with Original Vectors (Two-Stage Retrieval)
// ======================================================================================
// Binary search is fast but lossy. Best practice:
// 1. Use binary search to get top-K candidates (K=100-1000)
// 2. Rescore candidates with full-precision vectors
// 3. Return top-N results (N=10-50)
// ======================================================================================

// BinarySearchResult holds a candidate from binary search
type BinarySearchResult struct {
	ID              uint64
	HammingDistance int
	ApproxScore     float32 // Approximate cosine from Hamming
}

// RescoreCandidate holds a rescored result
type RescoreCandidate struct {
	ID         uint64
	FinalScore float32 // True cosine similarity
}

// BinaryIndex holds binary-quantized vectors for fast search
type BinaryIndex struct {
	quantizer   *BinaryQuantizer
	binaryData  []byte              // All binary vectors concatenated
	idMap       []uint64            // Maps index position to vector ID
	bytesPerVec int
}

// NewBinaryIndex creates a binary index
func NewBinaryIndex(dim int) *BinaryIndex {
	q := NewBinaryQuantizer(dim)
	return &BinaryIndex{
		quantizer:   q,
		bytesPerVec: q.BytesPerVector(),
	}
}

// Train trains the quantizer on sample vectors
func (idx *BinaryIndex) Train(vectors []float32) error {
	return idx.quantizer.Train(vectors)
}

// Add adds vectors to the index
func (idx *BinaryIndex) Add(id uint64, vector []float32) error {
	binary, err := idx.quantizer.Quantize(vector)
	if err != nil {
		return err
	}
	
	idx.binaryData = append(idx.binaryData, binary...)
	idx.idMap = append(idx.idMap, id)
	return nil
}

// AddBatch adds multiple vectors efficiently
func (idx *BinaryIndex) AddBatch(ids []uint64, vectors []float32) error {
	if len(vectors)%idx.quantizer.dim != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension")
	}
	
	binary, err := idx.quantizer.Quantize(vectors)
	if err != nil {
		return err
	}
	
	idx.binaryData = append(idx.binaryData, binary...)
	idx.idMap = append(idx.idMap, ids...)
	return nil
}

// Search finds top-k candidates using Hamming distance
// Uses parallel processing for large indexes
func (idx *BinaryIndex) Search(query []float32, k int) ([]BinarySearchResult, error) {
	// Quantize query
	binaryQuery, err := idx.quantizer.Quantize(query)
	if err != nil {
		return nil, err
	}
	
	numVectors := len(idx.idMap)
	if numVectors == 0 {
		return nil, nil
	}
	
	// For small indexes, use simple search
	if numVectors < 10000 {
		return idx.searchSimple(binaryQuery, k)
	}
	
	// Parallel search for large indexes
	return idx.searchParallel(binaryQuery, k)
}

func (idx *BinaryIndex) searchSimple(binaryQuery []byte, k int) ([]BinarySearchResult, error) {
	distances := HammingDistanceBatch(binaryQuery, idx.binaryData, idx.bytesPerVec)
	return idx.selectTopK(distances, k), nil
}

func (idx *BinaryIndex) searchParallel(binaryQuery []byte, k int) ([]BinarySearchResult, error) {
	numVectors := len(idx.idMap)
	numWorkers := runtime.NumCPU()
	chunkSize := (numVectors + numWorkers - 1) / numWorkers
	
	// Each worker finds local top-k
	type localResult struct {
		results []BinarySearchResult
	}
	
	var wg sync.WaitGroup
	localResults := make([]localResult, numWorkers)
	
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * chunkSize
			end := start + chunkSize
			if end > numVectors {
				end = numVectors
			}
			if start >= end {
				return
			}
			
			// Compute distances for this chunk
			chunkData := idx.binaryData[start*idx.bytesPerVec : end*idx.bytesPerVec]
			distances := HammingDistanceBatch(binaryQuery, chunkData, idx.bytesPerVec)
			
			// Convert to absolute indices
			for i := range distances {
				distances[i] = distances[i] // distance stays same
			}
			
			// Select local top-k
			results := make([]BinarySearchResult, 0, k)
			type idxDist struct {
				idx  int
				dist int
			}
			pairs := make([]idxDist, len(distances))
			for i, d := range distances {
				pairs[i] = idxDist{idx: start + i, dist: d}
			}
			
			for i := 0; i < k && i < len(pairs); i++ {
				minIdx := i
				for j := i + 1; j < len(pairs); j++ {
					if pairs[j].dist < pairs[minIdx].dist {
						minIdx = j
					}
				}
				pairs[i], pairs[minIdx] = pairs[minIdx], pairs[i]
				
				results = append(results, BinarySearchResult{
					ID:              idx.idMap[pairs[i].idx],
					HammingDistance: pairs[i].dist,
					ApproxScore:     HammingToCosineSimilarity(pairs[i].dist, idx.quantizer.dim),
				})
			}
			
			localResults[workerID] = localResult{results: results}
		}(w)
	}
	
	wg.Wait()
	
	// Merge local results
	allCandidates := make([]BinarySearchResult, 0, numWorkers*k)
	for _, lr := range localResults {
		allCandidates = append(allCandidates, lr.results...)
	}
	
	// Final top-k selection
	for i := 0; i < k && i < len(allCandidates); i++ {
		minIdx := i
		for j := i + 1; j < len(allCandidates); j++ {
			if allCandidates[j].HammingDistance < allCandidates[minIdx].HammingDistance {
				minIdx = j
			}
		}
		allCandidates[i], allCandidates[minIdx] = allCandidates[minIdx], allCandidates[i]
	}
	
	if k > len(allCandidates) {
		k = len(allCandidates)
	}
	return allCandidates[:k], nil
}

func (idx *BinaryIndex) selectTopK(distances []int, k int) []BinarySearchResult {
	type idxDist struct {
		idx  int
		dist int
	}
	pairs := make([]idxDist, len(distances))
	for i, d := range distances {
		pairs[i] = idxDist{idx: i, dist: d}
	}
	
	results := make([]BinarySearchResult, 0, k)
	for i := 0; i < k && i < len(pairs); i++ {
		minIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[minIdx].dist {
				minIdx = j
			}
		}
		pairs[i], pairs[minIdx] = pairs[minIdx], pairs[i]
		
		results = append(results, BinarySearchResult{
			ID:              idx.idMap[pairs[i].idx],
			HammingDistance: pairs[i].dist,
			ApproxScore:     HammingToCosineSimilarity(pairs[i].dist, idx.quantizer.dim),
		})
	}
	return results
}

// Size returns number of vectors in index
func (idx *BinaryIndex) Size() int {
	return len(idx.idMap)
}

// MemoryUsage returns approximate memory usage in bytes
func (idx *BinaryIndex) MemoryUsage() int {
	return len(idx.binaryData) + len(idx.idMap)*8
}
