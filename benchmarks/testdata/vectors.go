package testdata

import (
	"math"
	"math/rand"
)

// GenerateCentroids creates cluster centroids on the unit sphere.
// These centroids are shared between data vectors and query vectors
// to ensure queries have nearby neighbors in the dataset.
func GenerateCentroids(dim, clusters int, rng *rand.Rand) [][]float32 {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	centroids := make([][]float32, clusters)
	for c := 0; c < clusters; c++ {
		centroid := make([]float32, dim)
		var norm float64
		for d := 0; d < dim; d++ {
			v := rng.NormFloat64()
			centroid[d] = float32(v)
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for d := 0; d < dim; d++ {
			centroid[d] /= float32(norm)
		}
		centroids[c] = centroid
	}
	return centroids
}

// GenerateClusteredVectors creates vectors clustered around k centroids.
// Unlike uniform random vectors, clustered vectors create realistic
// distance distributions where nearest-neighbor search is non-trivial.
func GenerateClusteredVectors(count, dim, clusters int, spread float64, rng *rand.Rand) [][]float32 {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	if clusters <= 0 {
		clusters = 1
	}
	if clusters > count {
		clusters = count
	}

	centroids := GenerateCentroids(dim, clusters, rng)
	return generateFromCentroids(count, dim, centroids, spread, rng)
}

// GenerateClusteredDataset generates both data vectors and query vectors
// that share the same cluster centroids. This ensures queries have
// meaningful nearest neighbors in the dataset.
func GenerateClusteredDataset(dataCount, queryCount, dim, clusters int, spread float64, seed int64) (data, queries [][]float32) {
	// Use separate RNG for centroids to keep them stable
	centroidRng := rand.New(rand.NewSource(seed))
	centroids := GenerateCentroids(dim, clusters, centroidRng)

	// Data vectors
	dataRng := rand.New(rand.NewSource(seed + 1))
	data = generateFromCentroids(dataCount, dim, centroids, spread, dataRng)

	// Query vectors from same centroids but tighter spread
	queryRng := rand.New(rand.NewSource(seed + 2))
	queries = generateFromCentroids(queryCount, dim, centroids, spread*0.8, queryRng)

	return data, queries
}

func generateFromCentroids(count, dim int, centroids [][]float32, spread float64, rng *rand.Rand) [][]float32 {
	clusters := len(centroids)
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		c := i % clusters
		vec := make([]float32, dim)
		var norm float64
		for d := 0; d < dim; d++ {
			v := float64(centroids[c][d]) + rng.NormFloat64()*spread
			vec[d] = float32(v)
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for d := 0; d < dim; d++ {
				vec[d] /= float32(norm)
			}
		}
		vectors[i] = vec
	}
	return vectors
}

// GenerateQueries generates query vectors from the same distribution as the data.
// DEPRECATED: Use GenerateClusteredDataset instead for proper centroid sharing.
func GenerateQueries(count, dim, clusters int, spread float64, rng *rand.Rand) [][]float32 {
	return GenerateClusteredVectors(count, dim, clusters, spread*0.8, rng)
}

// GenerateUniformVectors generates uniformly random unit vectors.
// Included for comparison — these produce artificially easy benchmarks
// due to concentration of measure in high dimensions.
func GenerateUniformVectors(count, dim int, rng *rand.Rand) [][]float32 {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		var norm float64
		for d := 0; d < dim; d++ {
			v := rng.NormFloat64()
			vec[d] = float32(v)
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for d := 0; d < dim; d++ {
			vec[d] /= float32(norm)
		}
		vectors[i] = vec
	}
	return vectors
}

// ScaleConfig holds parameters for benchmark scale tiers.
type ScaleConfig struct {
	Name     string
	Count    int
	Clusters int
	Spread   float64
}

// StandardScales returns the canonical scale tiers for benchmarks.
// Short mode uses only Small; full mode uses all.
func StandardScales(short bool) []ScaleConfig {
	if short {
		return []ScaleConfig{
			{Name: "100K", Count: 100_000, Clusters: 50, Spread: 0.15},
		}
	}
	return []ScaleConfig{
		{Name: "100K", Count: 100_000, Clusters: 50, Spread: 0.15},
		{Name: "500K", Count: 500_000, Clusters: 100, Spread: 0.12},
		{Name: "1M", Count: 1_000_000, Clusters: 200, Spread: 0.10},
	}
}
