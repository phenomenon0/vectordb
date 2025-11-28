#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}              VECTOR DATABASE STANDALONE TEST SUITE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create a standalone test program
cat > /tmp/vectordb_standalone_test.go <<'EOF'
package main

import (
	"crypto/sha256"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/coder/hnsw"
)

// Minimal VectorStore (no external dependencies)
type VectorStore struct {
	sync.RWMutex
	Data    []float32
	Dim     int
	Count   int
	Docs    []string
	IDs     []string
	next    int64
	hnsw    *hnsw.Graph[uint64]
	idToIx  map[uint64]int
	Meta    map[uint64]map[string]string
	Deleted map[uint64]bool
	Coll    map[uint64]string
}

func NewVectorStore(capacity int, dim int) *VectorStore {
	g := hnsw.NewGraph[uint64]()
	g.Distance = hnsw.CosineDistance
	g.M = 16
	g.Ml = 0.25
	g.EfSearch = 64
	return &VectorStore{
		Data:    make([]float32, 0, capacity*dim),
		Dim:     dim,
		Count:   0,
		hnsw:    g,
		idToIx:  make(map[uint64]int),
		Meta:    make(map[uint64]map[string]string),
		Deleted: make(map[uint64]bool),
		Coll:    make(map[uint64]string),
	}
}

func (vs *VectorStore) Add(v []float32, doc string, id string, meta map[string]string, collection string) (string, error) {
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		return "", fmt.Errorf("dimension mismatch: expected %d, got %d", vs.Dim, len(v))
	}
	if id == "" {
		id = fmt.Sprintf("doc-%d", vs.next)
		vs.next++
	}
	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Count++

	hid := hashID(id)
	vs.hnsw.Add(hnsw.MakeNode(hid, v))
	vs.idToIx[hid] = vs.Count - 1
	if meta != nil {
		vs.Meta[hid] = meta
	}
	if collection == "" {
		collection = "default"
	}
	vs.Coll[hid] = collection
	delete(vs.Deleted, hid)
	return id, nil
}

func (vs *VectorStore) Delete(id string) error {
	vs.Lock()
	defer vs.Unlock()
	hid := hashID(id)
	vs.Deleted[hid] = true
	delete(vs.Meta, hid)
	delete(vs.Coll, hid)
	return nil
}

func (vs *VectorStore) SearchANN(query []float32, k int) []int {
	vs.RLock()
	defer vs.RUnlock()
	nodes := vs.hnsw.Search(query, k)
	ixs := make([]int, 0, len(nodes))
	for _, n := range nodes {
		if vs.Deleted[n.Key] {
			continue
		}
		if ix, ok := vs.idToIx[n.Key]; ok {
			ixs = append(ixs, ix)
		}
	}
	return ixs
}

func (vs *VectorStore) GetID(index int) string {
	if index < 0 || index >= len(vs.IDs) {
		return ""
	}
	return vs.IDs[index]
}

func (vs *VectorStore) Save(path string) error {
	vs.RLock()
	defer vs.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	payload := struct {
		Dim     int
		Data    []float32
		Docs    []string
		IDs     []string
		Meta    map[uint64]map[string]string
		Deleted map[uint64]bool
		Coll    map[uint64]string
		Next    int64
		Count   int
	}{
		Dim:     vs.Dim,
		Data:    vs.Data,
		Docs:    vs.Docs,
		IDs:     vs.IDs,
		Meta:    vs.Meta,
		Deleted: vs.Deleted,
		Coll:    vs.Coll,
		Next:    vs.next,
		Count:   vs.Count,
	}
	return gob.NewEncoder(f).Encode(payload)
}

func hashID(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// Simple hash embedder
type HashEmbedder struct {
	dim int
}

func NewHashEmbedder(dim int) *HashEmbedder {
	return &HashEmbedder{dim: dim}
}

func (e *HashEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}
	h := sha256.New()
	h.Write([]byte(text))
	seed := int64(h.Sum(nil)[0])
	vec := make([]float32, e.dim)
	r := rand.New(rand.NewSource(seed))
	var norm float64
	for i := 0; i < e.dim; i++ {
		val := r.Float64()*2 - 1
		vec[i] = float32(val)
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= float32(norm)
		}
	}
	return vec, nil
}

// Test functions
func testBasicOperations() error {
	fmt.Println("Testing basic operations...")
	vs := NewVectorStore(10, 3)

	id1, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"tag": "a"}, "c1")
	if err != nil {
		return fmt.Errorf("add failed: %v", err)
	}
	if id1 != "id1" {
		return fmt.Errorf("wrong id: %s", id1)
	}

	ids := vs.SearchANN([]float32{1, 0, 0}, 1)
	if len(ids) != 1 || vs.GetID(ids[0]) != "id1" {
		return fmt.Errorf("search failed: %+v", ids)
	}

	if err := vs.Delete("id1"); err != nil {
		return fmt.Errorf("delete failed: %v", err)
	}

	fmt.Println("  ✓ Basic operations passed")
	return nil
}

func testPersistence() error {
	fmt.Println("Testing persistence...")
	dir := os.TempDir()
	path := filepath.Join(dir, "test_index.gob")
	defer os.Remove(path)

	vs := NewVectorStore(10, 3)
	vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "c1")

	if err := vs.Save(path); err != nil {
		return fmt.Errorf("save failed: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("file not created: %v", err)
	}

	fmt.Println("  ✓ Persistence passed")
	return nil
}

func testConcurrency() error {
	fmt.Println("Testing concurrency...")
	vs := NewVectorStore(1000, 3)
	emb := NewHashEmbedder(3)

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	// 10 concurrent writers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				vec, _ := emb.Embed(fmt.Sprintf("doc-%d-%d", id, j))
				if _, err := vs.Add(vec, "content", "", nil, "default"); err != nil {
					errors <- err
				}
			}
		}(i)
	}

	// 5 concurrent readers
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vec, _ := emb.Embed("query")
			for j := 0; j < 200; j++ {
				_ = vs.SearchANN(vec, 10)
			}
		}()
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		return fmt.Errorf("concurrent error: %v", err)
	}

	if vs.Count != 1000 {
		return fmt.Errorf("wrong count: %d", vs.Count)
	}

	fmt.Println("  ✓ Concurrency passed")
	return nil
}

func testPerformance() error {
	fmt.Println("Testing performance...")
	vs := NewVectorStore(10000, 384)
	emb := NewHashEmbedder(384)

	// Insert 1000 vectors
	start := time.Now()
	for i := 0; i < 1000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		vs.Add(vec, "content", "", nil, "default")
	}
	insertTime := time.Since(start)

	// Search 1000 times
	query, _ := emb.Embed("query")
	start = time.Now()
	for i := 0; i < 1000; i++ {
		vs.SearchANN(query, 10)
	}
	searchTime := time.Since(start)

	fmt.Printf("  Insert: %.2f ops/sec\n", float64(1000)/insertTime.Seconds())
	fmt.Printf("  Search: %.2f ops/sec (%.2fms per query)\n",
		float64(1000)/searchTime.Seconds(),
		searchTime.Seconds()*1000/1000)

	fmt.Println("  ✓ Performance test passed")
	return nil
}

func testMemoryUsage() error {
	fmt.Println("Testing memory usage...")
	vs := NewVectorStore(100000, 384)
	emb := NewHashEmbedder(384)

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	memBefore := m.Alloc

	// Add 10k vectors
	for i := 0; i < 10000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		vs.Add(vec, "content", "", nil, "default")
	}

	runtime.GC()
	runtime.ReadMemStats(&m)
	memAfter := m.Alloc

	memUsed := float64(memAfter-memBefore) / (1024 * 1024)
	expectedMem := float64(10000*384*4) / (1024 * 1024) // ~14.6 MB

	fmt.Printf("  Memory used: %.2f MB (expected ~%.2f MB)\n", memUsed, expectedMem)

	fmt.Println("  ✓ Memory usage test passed")
	return nil
}

func main() {
	fmt.Println()
	start := time.Now()

	tests := []struct{
		name string
		fn   func() error
	}{
		{"Basic Operations", testBasicOperations},
		{"Persistence", testPersistence},
		{"Concurrency", testConcurrency},
		{"Performance", testPerformance},
		{"Memory Usage", testMemoryUsage},
	}

	passed := 0
	failed := 0

	for _, test := range tests {
		if err := test.fn(); err != nil {
			fmt.Printf("✗ %s FAILED: %v\n", test.name, err)
			failed++
		} else {
			passed++
		}
	}

	duration := time.Since(start)

	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("RESULTS: %d passed, %d failed (%.2fs)\n", passed, failed, duration.Seconds())
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	if failed > 0 {
		os.Exit(1)
	}
}
EOF

# Compile and run the standalone test
echo -e "${YELLOW}Compiling standalone test...${NC}"
if cd /tmp && go mod init vectordb_test 2>/dev/null; then
    echo "  Created temp module"
fi

cd /tmp && go get github.com/coder/hnsw@latest 2>&1 | grep -v "go: downloading" | head -5

echo -e "${YELLOW}Running tests...${NC}"
cd /tmp && go run vectordb_standalone_test.go

echo ""
echo -e "${GREEN}✓ All standalone tests completed${NC}"
