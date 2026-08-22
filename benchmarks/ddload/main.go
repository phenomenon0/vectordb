// Command ddload measures DeepData canonical V3 throughput with an efficient
// Go client, removing Python-client serialization noise from the measurement.
//
// Usage: go run ./benchmarks/ddload -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sync"
	"strconv"
	"sync/atomic"
	"time"
)

const (
	baseURL   = "http://127.0.0.1:8093"
	apiToken  = "ddload-token-0123456789abcdef0123456789abcdef"
	tenant    = "bench"
	coll      = "bench"
	batchSize = 2000
	workers   = 4
)

func authReq(method, url string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest(method, baseURL+url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiToken)
	req.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(req)
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

// appendFloat appends the shortest round-tripping decimal form of f. Integral
// values take the integer form ("1" not "1e+00"), matching Go's json encoder.
func appendFloat(dst []byte, f float64) []byte {
	if f == math.Trunc(f) && math.Abs(f) < 1e15 {
		return strconv.AppendInt(dst, int64(f), 10)
	}
	return strconv.AppendFloat(dst, f, 'g', -1, 64)
}

func main() {
	basePath := flag.String("base", "", "path to .fvecs base vectors")
	flag.Parse()
	if *basePath == "" {
		flag.Usage()
		os.Exit(2)
	}

	vectors, err := readFvecs(*basePath, 100000)
	if err != nil {
		fmt.Println("read fvecs:", err)
		os.Exit(1)
	}
	fmt.Printf("loaded %d vectors (%dd)\n", len(vectors), len(vectors[0]))

	client := &http.Client{Timeout: 120 * time.Second}

	// Create collection with HNSW matching the Python harness schema.
	schema := fmt.Sprintf(`{"name":%q,"fields":[{"name":"embedding","type":"dense","dim":%d,"index":{"type":"hnsw","params":{"m":16,"ef_construction":200}}}]}`,
		coll, len(vectors[0]))
	createResp, err := authReq("POST", "/v3/tenants/"+tenant+"/collections", []byte(schema))
	must(err)
	if createResp.StatusCode != 201 && createResp.StatusCode != 200 {
		b, _ := io.ReadAll(createResp.Body)
		fmt.Printf("create collection: %d %s\n", createResp.StatusCode, b)
		os.Exit(1)
	}
	io.Copy(io.Discard, createResp.Body)
	createResp.Body.Close()

	var next uint64 = 1 // canonical IDs are caller-supplied; 0 = auto-assign
	start := time.Now()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	errCh := make(chan error, workers)

	encodeBatch := func(from, to int) []byte {
		buf := make([]byte, 0, (to-from)*(len(vectors[0])*10+40))
		buf = append(buf, `{"documents":[`...)
		for i := from; i < to; i++ {
			if i > from {
				buf = append(buf, ',')
			}
			id := uint64(i) + 1
			buf = append(buf, `{"id":`...)
			buf = strconv.AppendUint(buf, id, 10)
			buf = append(buf, `,"vectors":{"embedding":`...)
			buf = appendVectorJSON(buf, vectors[i])
			buf = append(buf, `}}`...)
		}
		return append(buf, ']', '}')
	}

	total := int64(len(vectors))
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				n := atomic.AddUint64(&next, uint64(batchSize))
				from := int(n) - batchSize
				if from >= int(total) {
					return
				}
				to := from + batchSize
				if to > int(total) {
					to = int(total)
				}
				body := encodeBatch(from, to)
				req, err := http.NewRequest("POST",
					baseURL+"/v3/tenants/"+tenant+"/collections/"+coll+"/docs/batch",
					bytes.NewReader(body))
				if err != nil {
					errCh <- err
					return
				}
				req.Header.Set("Authorization", "Bearer "+apiToken)
				req.Header.Set("Content-Type", "application/json")
				resp, err := client.Do(req)
				if err != nil {
					errCh <- err
					return
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode != 200 && resp.StatusCode != 201 {
					errCh <- fmt.Errorf("batch insert %d-%d: status %d", from, to, resp.StatusCode)
					return
				}
			}
		}()
	}

	wg.Wait()
	close(stop)
	select {
	case e := <-errCh:
		fmt.Println("worker error:", e)
		os.Exit(1)
	default:
	}

	elapsed := time.Since(start)
	qps := float64(total) / elapsed.Seconds()
	fmt.Printf("INSERT: %d vectors in %.2fs = %.0f vec/s (%.0f docs/s per worker avg)\n",
		total, elapsed.Seconds(), qps, qps/float64(workers))

	// ── Search phase: serial + concurrent ──
	query := vectors[len(vectors)-1]
	body := buildSearchBody(query)

	// Warmup
	for i := 0; i < 20; i++ {
		resp, err := authReq("POST", "/v3/tenants/"+tenant+"/collections/"+coll+"/search", body)
		must(err)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}

	n := 500
	t0 := time.Now()
	for i := 0; i < n; i++ {
		resp, err := authReq("POST", "/v3/tenants/"+tenant+"/collections/"+coll+"/search", body)
		must(err)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
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
				resp, err := authReq("POST", "/v3/tenants/"+tenant+"/collections/"+coll+"/search", body)
				if err == nil {
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
			}
		}()
	}
	swg.Wait()
	conc := time.Since(t1)
	fmt.Printf("SEARCH concurrent(8T): %.0f qps\n", float64(counter)/conc.Seconds())

	// Cleanup
	delResp, err := authReq("DELETE", "/v3/tenants/"+tenant+"/collections/"+coll, nil)
	if err == nil {
		io.Copy(io.Discard, delResp.Body)
		delResp.Body.Close()
	}
}

func buildSearchBody(query []float32) []byte {
	buf := make([]byte, 0, len(query)*10+64)
	buf = append(buf, `{"queries":{"embedding":`...)
	buf = appendVectorJSON(buf, query)
	buf = append(buf, `},"top_k":10}`...)
	return buf
}

func must(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
