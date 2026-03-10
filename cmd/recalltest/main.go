package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/coder/hnsw"
	"github.com/phenomenon0/vectordb/internal/index/simd"
)

func main() {
	rng := rand.New(rand.NewSource(42))
	dist := simd.CosineDistanceF32

	for _, dim := range []int{128, 768, 1536} {
		for _, n := range []int{1000, 10000, 50000} {
			testRecall(rng, dist, dim, n)
		}
	}
}

func testRecall(rng *rand.Rand, dist hnsw.DistanceFunc, dim, n int) {
	nq := 100
	k := 10

	vectors := make([][]float32, n)
	for i := range vectors {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		var norm float32
		for _, x := range v {
			norm += x * x
		}
		norm = float32(math.Sqrt(float64(norm)))
		for j := range v {
			v[j] /= norm
		}
		vectors[i] = v
	}

	queries := make([][]float32, nq)
	for i := range queries {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		var norm float32
		for _, x := range v {
			norm += x * x
		}
		norm = float32(math.Sqrt(float64(norm)))
		for j := range v {
			v[j] /= norm
		}
		queries[i] = v
	}

	// Brute force ground truth
	groundTruth := make([][]int, nq)
	for qi := 0; qi < nq; qi++ {
		type idDist struct {
			id   int
			dist float32
		}
		dists := make([]idDist, n)
		for i := 0; i < n; i++ {
			dists[i] = idDist{i, dist(queries[qi], vectors[i])}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
		gt := make([]int, k)
		for i := 0; i < k; i++ {
			gt[i] = dists[i].id
		}
		groundTruth[qi] = gt
	}

	g := hnsw.NewGraph[uint64]()
	g.Distance = dist
	g.M = 16
	g.Ml = 1.0 / math.Log(float64(16))
	g.EfSearch = 200
	g.Rng = rand.New(rand.NewSource(12345))

	t0 := time.Now()
	for i := 0; i < n; i++ {
		g.Add(hnsw.MakeNode(uint64(i), vectors[i]))
	}
	buildTime := time.Since(t0)

	// Test with different ef values
	fmt.Printf("\n%dd-%dk  build=%v\n", dim, n/1000, buildTime)
	for _, ef := range []int{200, 500, 1000} {
		if ef > n {
			continue
		}
		totalRecall := 0.0
		for qi := 0; qi < nq; qi++ {
			results := g.SearchWithEf(queries[qi], k, ef, nil)
			gtSet := make(map[uint64]bool)
			for _, id := range groundTruth[qi] {
				gtSet[uint64(id)] = true
			}
			hits := 0
			for _, r := range results {
				if gtSet[r.Key] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		fmt.Printf("  ef=%4d: recall@%d = %.4f\n", ef, k, totalRecall/float64(nq))
	}
}
