// Command ddload-qdrant measures Qdrant with the same efficient-Go-client
// methodology as ./ddload, so read-side comparisons are server-bound rather
// than Python-client-bound.
//
// Usage: go run ./benchmarks/ddload-qdrant -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	baseURL   = "http://127.0.0.1:6333"
	coll      = "bench"
	batchSize = 2000
	workers   = 4
)

// appendVectorJSON writes a JSON number array without reflection.
func appendVectorJSON(dst []byte, v []float32) []byte {
	dst = append(dst, '[')
	for i, x := range v {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendFloat(dst, float64(x))
	}
	return append(dst, ']')
}

func appendFloat(dst []byte, f float64) []byte {
	if f == math.Trunc(f) && math.Abs(f) < 1e15 {
		return strconv.AppendInt(dst, int64(f), 10)
	}
	return strconv.AppendFloat(dst, f, 'g', -1, 64)
}

// readFvecs parses .fvecs: each record is int32 dim + dim float32 values.
func readFvecs(path string, maxN int) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out [][]float32
	var header [4]byte
	for len(out) < maxN {
		if _, err := io.ReadFull(f, header[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		dim := int32(header[0]) | int32(header[1])<<8 | int32(header[2])<<16 | int32(header[3])<<24
		buf := make([]byte, int(dim)*4)
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, err
		}
		vec := make([]float32, dim)
		for i := range vec {
			bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 | uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
			vec[i] = math.Float32frombits(bits)
		}
		out = append(out, vec)
	}
	return out, nil
}

func must(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func main() {
	basePath := flag.String("base", "", "path to .fvecs base vectors")
	flag.Parse()
	if *basePath == "" {
		flag.Usage()
		os.Exit(2)
	}

	vectors, err := readFvecs(*basePath, 100000)
	must(err)
	fmt.Printf("loaded %d vectors (%dd)\n", len(vectors), len(vectors[0]))

	client := &http.Client{Timeout: 120 * time.Second}
	post := func(path string, body []byte) (*http.Response, error) {
		req, err := http.NewRequest("POST", baseURL+path, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		return client.Do(req)
	}
	put := func(path string, body []byte) (*http.Response, error) {
		req, err := http.NewRequest("PUT", baseURL+path, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		return client.Do(req)
	}

	// Recreate collection mirroring the recorded harness: cosine, m=16,
	// ef_construct=300 (dataset defaults in recall_test.py).
	delReq, _ := http.NewRequest("DELETE", baseURL+"/collections/"+coll, nil)
	if delResp, err := client.Do(delReq); err == nil {
		io.Copy(io.Discard, delResp.Body)
		delResp.Body.Close()
	}
	// indexing_threshold default (20MB) leaves small segments unindexed
	// forever on this dataset; set a sane per-collection value up front.
	schema := fmt.Sprintf(`{"vectors":{"size":%d,"distance":"Cosine"},"hnsw_config":{"m":16,"ef_construct":300},"optimizer_config":{"indexing_threshold":2048}}`, len(vectors[0]))
	resp, err := put("/collections/"+coll, []byte(schema))
	must(err)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Printf("create collection: %d %s\n", resp.StatusCode, body)
		os.Exit(1)
	}

	total := len(vectors)
	start := time.Now()
	errCh := make(chan error, workers)
	var wg sync.WaitGroup
	var next int64 = 0

	encodeBatch := func(from, to int) []byte {
		buf := make([]byte, 0, (to-from)*(len(vectors[0])*10+24))
		buf = append(buf, `{"batch":{"ids":[`...)
		for i := from; i < to; i++ {
			if i > from {
				buf = append(buf, ',')
			}
			buf = strconv.AppendUint(buf, uint64(i), 10)
		}
		buf = append(buf, `],"vectors":[`...)
		for i := from; i < to; i++ {
			if i > from {
				buf = append(buf, ',')
			}
			buf = appendVectorJSON(buf, vectors[i])
		}
		return append(buf, `]},"wait":true}`...)
	}

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				n := atomic.AddInt64(&next, int64(batchSize))
				from := int(n) - batchSize
				if from >= total {
					return
				}
				to := from + batchSize
				if to > total {
					to = total
				}
				payload := encodeBatch(from, to)
				r2, err := put("/collections/"+coll+"/points?wait=true", payload)
				if err != nil {
					errCh <- err
					return
				}
				io.Copy(io.Discard, r2.Body)
				r2.Body.Close()
				if r2.StatusCode != 200 {
					errCh <- fmt.Errorf("batch insert %d-%d: status %d", from, to, r2.StatusCode)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	failed := false
	for e := range errCh {
		if e != nil {
			failed = true
			fmt.Println("worker error:", e)
		}
	}
	if failed {
		os.Exit(1)
	}
	elapsed := time.Since(start)
	fmt.Printf("INSERT(ack): %d vectors in %.2fs = %.0f vec/s\n",
		total, elapsed.Seconds(), float64(total)/elapsed.Seconds())

	// Time-to-fully-searchable: HNSW builds asynchronously; poll the index
	// until every vector is actually indexed.
	idxStart := time.Now()
	for {
		idxed, err := getIndexedCount(client)
		must(err)
		if idxed >= total {
			break
		}
		if time.Since(idxStart) > 30*time.Minute {
			fmt.Println("indexing did not complete in 30m")
			os.Exit(1)
		}
		time.Sleep(500 * time.Millisecond)
	}
	catchup := time.Since(idxStart)
	fmt.Printf("OPTIMIZER catch-up: %.2fs (time-to-searchable %.2fs total)\n",
		catchup.Seconds(), time.Since(start).Seconds())

	query := vectors[len(vectors)-1]
	searchBody := buildSearchBody(query, 10)

	for i := 0; i < 20; i++ { // warmup
		r3, err := post("/collections/"+coll+"/points/search", searchBody)
		must(err)
		io.Copy(io.Discard, r3.Body)
		r3.Body.Close()
	}

	n := 500
	t0 := time.Now()
	for i := 0; i < n; i++ {
		r3, err := post("/collections/"+coll+"/points/search", searchBody)
		must(err)
		io.Copy(io.Discard, r3.Body)
		r3.Body.Close()
	}
	serial := time.Since(t0)
	fmt.Printf("SEARCH serial: %.0f qps (%.2f ms avg)\n",
		float64(n)/serial.Seconds(), float64(serial.Microseconds())/float64(n)/1000)

	var counter int64
	t1 := time.Now()
	var swg sync.WaitGroup
	for w := 0; w < 8; w++ {
		swg.Add(1)
		go func() {
			defer swg.Done()
			for {
				if atomic.LoadInt64(&counter) >= 4000 {
					return
				}
				atomic.AddInt64(&counter, 1)
				r3, err := post("/collections/"+coll+"/points/search", searchBody)
				if err == nil {
					io.Copy(io.Discard, r3.Body)
					r3.Body.Close()
				}
			}
		}()
	}
	swg.Wait()
	conc := time.Since(t1)
	fmt.Printf("SEARCH concurrent(8T): %.0f qps\n", float64(counter)/conc.Seconds())
}

func getIndexedCount(client *http.Client) (int, error) {
	resp, err := client.Get(baseURL + "/collections/" + coll)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var parsed struct {
		Result struct {
			IndexedVectorsCount int `json:"indexed_vectors_count"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return 0, err
	}
	return parsed.Result.IndexedVectorsCount, nil
}

func buildSearchBody(query []float32, limit int) []byte {
	buf := make([]byte, 0, len(query)*10+64)
	buf = append(buf, `{"vector":`...)
	buf = appendVectorJSON(buf, query)
	buf = append(buf, `,"limit":`...)
	buf = strconv.AppendInt(buf, int64(limit), 10)
	return append(buf, '}')
}
