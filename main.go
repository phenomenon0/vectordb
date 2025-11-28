package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"agentscope/core"
	"agentscope/tools"
	"github.com/coder/hnsw"
)

// ======================================================================================
// Vector Store (HNSW + metadata + persistence)
// ======================================================================================

type VectorStore struct {
	sync.RWMutex
	Data        []float32
	Dim         int
	Count       int
	Docs        []string
	IDs         []string
	next        int64
	hnsw        *hnsw.Graph[uint64]
	idToIx      map[uint64]int
	Meta        map[uint64]map[string]string
	Deleted     map[uint64]bool
	Coll        map[uint64]string
	NumMeta     map[uint64]map[string]float64
	TimeMeta    map[uint64]map[string]time.Time
	walPath     string
	walMu       sync.Mutex
	walMaxBytes int64
	walMaxOps   int
	walOps      int
	apiToken    string
	rl          *rateLimiter
	checksum    string
	lastSaved   time.Time
	// Lexical stats for hybrid/BM25
	lexTF   map[uint64]map[string]int
	docLen  map[uint64]int
	df      map[string]int
	sumDocL int
}

func NewVectorStore(capacity int, dim int) *VectorStore {
	cfg := loadHNSWConfig()
	g := hnsw.NewGraph[uint64]()
	g.Distance = hnsw.CosineDistance
	g.M = cfg.M
	g.Ml = cfg.Ml
	g.EfSearch = cfg.EfSearch
	return &VectorStore{
		Data:        make([]float32, 0, capacity*dim),
		Dim:         dim,
		Count:       0,
		hnsw:        g,
		idToIx:      make(map[uint64]int),
		Meta:        make(map[uint64]map[string]string),
		Deleted:     make(map[uint64]bool),
		Coll:        make(map[uint64]string),
		NumMeta:     make(map[uint64]map[string]float64),
		TimeMeta:    make(map[uint64]map[string]time.Time),
		walMaxBytes: 0,
		walMaxOps:   0,
		apiToken:    os.Getenv("API_TOKEN"),
		lexTF:       make(map[uint64]map[string]int),
		docLen:      make(map[uint64]int),
		df:          make(map[string]int),
		sumDocL:     0,
	}
}

// Add appends a vector/doc/meta.
func (vs *VectorStore) Add(v []float32, doc string, id string, meta map[string]string, collection string) string {
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		panic("dimension mismatch")
	}
	if id == "" {
		id = fmt.Sprintf("doc-%d", vs.next)
		vs.next++
	}
	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Count++

	toks := tokenize(doc)
	vs.ingestLex(hashID(id), toks)

	hid := hashID(id)
	vs.hnsw.Add(hnsw.MakeNode(hid, v))
	vs.idToIx[hid] = vs.Count - 1
	if meta != nil {
		vs.Meta[hid] = meta
		vs.ingestMeta(hid, meta)
	}
	if collection == "" {
		collection = "default"
	}
	vs.Coll[hid] = collection
	delete(vs.Deleted, hid)
	vs.appendWAL("insert", id, doc, meta, v)
	return id
}

// Upsert replaces existing vector/doc/meta if ID exists; otherwise adds new.
func (vs *VectorStore) Upsert(v []float32, doc string, id string, meta map[string]string, collection string) string {
	if id == "" {
		return vs.Add(v, doc, id, meta, collection)
	}
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		panic("dimension mismatch")
	}
	hid := hashID(id)
	if ix, ok := vs.idToIx[hid]; ok && ix >= 0 && ix < len(vs.IDs) {
		vs.ejectLex(hid)
		vs.ejectMeta(hid)
		copy(vs.Data[ix*vs.Dim:(ix+1)*vs.Dim], v)
		vs.Docs[ix] = doc
		vs.ingestLex(hid, tokenize(doc))
		if meta != nil {
			vs.Meta[hid] = meta
			vs.ingestMeta(hid, meta)
		} else {
			delete(vs.Meta, hid)
			vs.ejectMeta(hid)
		}
		if collection != "" {
			vs.Coll[hid] = collection
		}
		delete(vs.Deleted, hid)
		vs.appendWAL("upsert", id, doc, meta, v)
		return id
	}

	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Count++
	vs.hnsw.Add(hnsw.MakeNode(hid, v))
	vs.idToIx[hid] = vs.Count - 1
	vs.ingestLex(hid, tokenize(doc))
	if meta != nil {
		vs.Meta[hid] = meta
		vs.ingestMeta(hid, meta)
	}
	if collection == "" {
		collection = "default"
	}
	vs.Coll[hid] = collection
	delete(vs.Deleted, hid)
	vs.appendWAL("insert", id, doc, meta, v)
	return id
}

func (vs *VectorStore) Delete(id string) {
	vs.Lock()
	defer vs.Unlock()
	hid := hashID(id)
	vs.Deleted[hid] = true
	vs.ejectLex(hid)
	vs.ejectMeta(hid)
	delete(vs.Meta, hid)
	delete(vs.Coll, hid)
	vs.appendWAL("delete", id, "", nil, nil)
}

func (vs *VectorStore) Get(index int) []float32 {
	offset := index * vs.Dim
	return vs.Data[offset : offset+vs.Dim]
}

func (vs *VectorStore) GetDoc(index int) string {
	if index < 0 || index >= len(vs.Docs) {
		return ""
	}
	return vs.Docs[index]
}

func (vs *VectorStore) GetID(index int) string {
	if index < 0 || index >= len(vs.IDs) {
		return ""
	}
	return vs.IDs[index]
}

// Brute-force scan (used for debugging).
func (vs *VectorStore) Search(query []float32, k int) []int {
	vs.RLock()
	defer vs.RUnlock()
	if k <= 0 || vs.Count == 0 {
		return nil
	}
	bestIDs := make([]int, 0, k)
	bestScores := make([]float32, 0, k)
	for i := 0; i < vs.Count; i++ {
		hid := hashID(vs.IDs[i])
		if vs.Deleted[hid] {
			continue
		}
		vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]
		score := DotProduct(query, vec)
		if len(bestIDs) < k {
			bestIDs = append(bestIDs, i)
			bestScores = append(bestScores, score)
			continue
		}
		minIdx := 0
		for j := 1; j < k; j++ {
			if bestScores[j] < bestScores[minIdx] {
				minIdx = j
			}
		}
		if score > bestScores[minIdx] {
			bestScores[minIdx] = score
			bestIDs[minIdx] = i
		}
	}
	return bestIDs
}

// ANN search via HNSW.
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

// Hybrid score combines cosine and BM25-like lexical score.
func (vs *VectorStore) hybridScore(hid uint64, qVec []float32, qTokens []string, alpha float64) float64 {
	vecScore := float64(0)
	if ix, ok := vs.idToIx[hid]; ok {
		dVec := vs.Data[ix*vs.Dim : (ix+1)*vs.Dim]
		vecScore = float64(DotProduct(qVec, dVec))
	}
	bm := vs.bm25(hid, qTokens)
	return alpha*vecScore + (1-alpha)*bm
}

func (vs *VectorStore) ingestLex(hid uint64, toks []string) {
	if len(toks) == 0 {
		return
	}
	tf := make(map[string]int)
	seen := make(map[string]bool)
	for _, t := range toks {
		if t == "" {
			continue
		}
		tf[t]++
		if !seen[t] {
			vs.df[t]++
			seen[t] = true
		}
	}
	prevLen := vs.docLen[hid]
	vs.sumDocL += len(toks) - prevLen
	vs.docLen[hid] = len(toks)
	vs.lexTF[hid] = tf
}

func (vs *VectorStore) ejectLex(hid uint64) {
	tf, ok := vs.lexTF[hid]
	if !ok {
		return
	}
	for term := range tf {
		if vs.df[term] > 0 {
			vs.df[term]--
		}
	}
	vs.sumDocL -= vs.docLen[hid]
	delete(vs.lexTF, hid)
	delete(vs.docLen, hid)
}

func (vs *VectorStore) bm25(hid uint64, qTokens []string) float64 {
	activeDocs := vs.Count - len(vs.Deleted)
	if activeDocs <= 0 {
		return 0
	}
	tf := vs.lexTF[hid]
	if tf == nil {
		return 0
	}
	avgdl := float64(vs.sumDocL)
	if avgdl == 0 {
		avgdl = 1
	} else {
		avgdl = avgdl / float64(activeDocs)
	}
	// BM25 parameters
	k1 := 1.2
	b := 0.75
	score := 0.0
	seen := make(map[string]bool)
	for _, qt := range qTokens {
		if seen[qt] {
			continue
		}
		seen[qt] = true
		df := vs.df[qt]
		if df == 0 {
			continue
		}
		idf := math.Log((float64(activeDocs)-float64(df)+0.5)/(float64(df)+0.5) + 1)
		tfDoc := float64(tf[qt])
		if tfDoc == 0 {
			continue
		}
		num := tfDoc * (k1 + 1)
		den := tfDoc + k1*(1-b+b*float64(vs.docLen[hid])/avgdl)
		score += idf * (num / den)
	}
	return score
}

func (vs *VectorStore) ingestMeta(hid uint64, meta map[string]string) {
	if meta == nil {
		return
	}
	nums := make(map[string]float64)
	times := make(map[string]time.Time)
	for k, v := range meta {
		if v == "" {
			continue
		}
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			times[k] = t
			continue
		}
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			nums[k] = f
		}
	}
	if len(nums) > 0 {
		vs.NumMeta[hid] = nums
	}
	if len(times) > 0 {
		vs.TimeMeta[hid] = times
	}
}

func (vs *VectorStore) ejectMeta(hid uint64) {
	delete(vs.NumMeta, hid)
	delete(vs.TimeMeta, hid)
}

// Persistence snapshot.
func (vs *VectorStore) Save(path string) error {
	vs.RLock()
	defer vs.RUnlock()

	tmp := path + ".tmp"
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer f.Close()

	var hBuf []byte
	if vs.hnsw != nil {
		var buf bytes.Buffer
		if err := vs.hnsw.Export(&buf); err == nil {
			hBuf = buf.Bytes()
		}
	}

	payload := struct {
		Dim       int
		Data      []float32
		Docs      []string
		IDs       []string
		Meta      map[uint64]map[string]string
		Deleted   map[uint64]bool
		Coll      map[uint64]string
		Next      int64
		Count     int
		HNSW      []byte
		Checksum  string
		LastSaved time.Time
		LexTF     map[uint64]map[string]int
		DocLen    map[uint64]int
		DF        map[string]int
		SumDocL   int
		NumMeta   map[uint64]map[string]float64
		TimeMeta  map[uint64]map[string]time.Time
	}{
		Dim:       vs.Dim,
		Data:      vs.Data,
		Docs:      vs.Docs,
		IDs:       vs.IDs,
		Meta:      vs.Meta,
		Deleted:   vs.Deleted,
		Coll:      vs.Coll,
		Next:      vs.next,
		Count:     vs.Count,
		HNSW:      hBuf,
		Checksum:  vs.checksum,
		LastSaved: time.Now(),
		LexTF:     vs.lexTF,
		DocLen:    vs.docLen,
		DF:        vs.df,
		SumDocL:   vs.sumDocL,
		NumMeta:   vs.NumMeta,
		TimeMeta:  vs.TimeMeta,
	}
	if err := gob.NewEncoder(f).Encode(payload); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	vs.checksum = vs.computeChecksum()
	vs.lastSaved = payload.LastSaved
	if vs.walPath != "" {
		_ = os.Remove(vs.walPath)
	}
	return os.Rename(tmp, path)
}

// Load snapshot or init new store.
func loadOrInitStore(path string, capacity int, dim int) (*VectorStore, bool) {
	if _, err := os.Stat(path); err == nil {
		f, err := os.Open(path)
		if err != nil {
			fmt.Printf("warning: failed to open index, rebuilding: %v\n", err)
			return NewVectorStore(capacity, dim), false
		}
		defer f.Close()

		var payload struct {
			Dim       int
			Data      []float32
			Docs      []string
			IDs       []string
			Meta      map[uint64]map[string]string
			Deleted   map[uint64]bool
			Coll      map[uint64]string
			Next      int64
			Count     int
			HNSW      []byte
			Checksum  string
			LastSaved time.Time
			LexTF     map[uint64]map[string]int
			DocLen    map[uint64]int
			DF        map[string]int
			SumDocL   int
			NumMeta   map[uint64]map[string]float64
			TimeMeta  map[uint64]map[string]time.Time
		}
		if err := gob.NewDecoder(f).Decode(&payload); err != nil {
			fmt.Printf("warning: failed to decode index, rebuilding: %v\n", err)
			return NewVectorStore(capacity, dim), false
		}
		vs := &VectorStore{
			Data:        payload.Data,
			Dim:         payload.Dim,
			Count:       payload.Count,
			Docs:        payload.Docs,
			IDs:         payload.IDs,
			next:        payload.Next,
			Meta:        payload.Meta,
			Deleted:     payload.Deleted,
			Coll:        payload.Coll,
			hnsw:        hnsw.NewGraph[uint64](),
			idToIx:      make(map[uint64]int),
			walPath:     path + ".wal",
			walMu:       sync.Mutex{},
			walMaxBytes: 0,
			walMaxOps:   0,
			apiToken:    os.Getenv("API_TOKEN"),
			checksum:    payload.Checksum,
			lastSaved:   payload.LastSaved,
			lexTF:       payload.LexTF,
			docLen:      payload.DocLen,
			df:          payload.DF,
			sumDocL:     payload.SumDocL,
			NumMeta:     payload.NumMeta,
			TimeMeta:    payload.TimeMeta,
		}
		if vs.checksum == "" {
			vs.checksum = fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d", payload.Count, payload.Next)))
		}
		// Checksum is best-effort; warn but continue to allow recovery.
		if !vs.validateChecksum() {
			fmt.Printf("warning: checksum mismatch; continuing with loaded snapshot\n")
		}
		for i, idStr := range vs.IDs {
			vs.idToIx[hashID(idStr)] = i
		}
		if vs.Meta == nil {
			vs.Meta = make(map[uint64]map[string]string)
		}
		if vs.Deleted == nil {
			vs.Deleted = make(map[uint64]bool)
		}
		if vs.Coll == nil {
			vs.Coll = make(map[uint64]string)
		}
		if vs.NumMeta == nil {
			vs.NumMeta = make(map[uint64]map[string]float64)
		}
		if vs.TimeMeta == nil {
			vs.TimeMeta = make(map[uint64]map[string]time.Time)
		}
		if vs.lexTF == nil {
			vs.lexTF = make(map[uint64]map[string]int)
		}
		if vs.docLen == nil {
			vs.docLen = make(map[uint64]int)
		}
		if vs.df == nil {
			vs.df = make(map[string]int)
		}
		if len(payload.HNSW) > 0 {
			if err := vs.hnsw.Import(bytes.NewReader(payload.HNSW)); err != nil {
				fmt.Printf("warning: failed to load hnsw graph, rebuilding: %v\n", err)
				vs.hnsw = hnsw.NewGraph[uint64]()
			}
		}
		if vs.next == 0 {
			vs.next = int64(len(vs.IDs))
		}
		replayWAL(vs)
		return vs, true
	}
	vs := NewVectorStore(capacity, dim)
	vs.walPath = path + ".wal"
	return vs, false
}

type hnswConfig struct {
	M        int
	Ml       float64
	EfSearch int
}

func loadHNSWConfig() hnswConfig {
	cfg := hnswConfig{M: 16, Ml: 0.25, EfSearch: 64}
	if v := os.Getenv("HNSW_M"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.M = n
		}
	}
	if v := os.Getenv("HNSW_ML"); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil && n > 0 {
			cfg.Ml = n
		}
	}
	if v := os.Getenv("HNSW_EFSEARCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.EfSearch = n
		}
	}
	return cfg
}

// ======================================================================================
// Tools and Agents
// ======================================================================================

type RetrievalTool struct {
	Store    *VectorStore
	Embedder Embedder
}

func (t *RetrievalTool) Name() string { return "flat_buffer_retrieval" }

type RetrievalInput struct {
	Query string `json:"query"`
}

type RetrievalResult struct {
	IDs   []string `json:"ids"`
	Docs  []string `json:"docs"`
	Stats string   `json:"stats"`
}

func (t *RetrievalTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	var queryText string
	if ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*RetrievalInput); ok && input != nil {
			queryText = input.Query
		}
	}
	if queryText == "" {
		queryText = "default"
	}

	qVec, err := t.Embedder.Embed(queryText)
	if err != nil {
		return &core.ToolExecResult{
			Status: core.ToolFailed,
			Error:  fmt.Sprintf("embedder error: %v", err),
		}
	}

	start := time.Now()
	ids := t.Store.SearchANN(qVec, 3)
	elapsed := time.Since(start)

	output := fmt.Sprintf("Search completed in %s. Found IDs: %v.", elapsed, ids)
	docs := make([]string, 0, len(ids))
	idStrings := make([]string, 0, len(ids))
	for _, id := range ids {
		docs = append(docs, t.Store.GetDoc(id))
		idStrings = append(idStrings, t.Store.GetID(id))
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &RetrievalResult{
			IDs:   idStrings,
			Docs:  docs,
			Stats: output,
		},
		Duration: elapsed,
	}
}

type RerankInput struct {
	Query string   `json:"query"`
	Docs  []string `json:"docs"`
	TopK  int      `json:"top_k"`
}

type RerankResult struct {
	Docs   []string  `json:"docs"`
	Scores []float32 `json:"scores"`
	Stats  string    `json:"stats"`
	IDs    []int     `json:"ids,omitempty"`
}

type RerankTool struct {
	Reranker Reranker
}

func (t *RerankTool) Name() string { return "reranker" }

func (t *RerankTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	var input RerankInput
	if ctx.Request != nil && ctx.Request.ToolReq != nil {
		if in, ok := ctx.Request.ToolReq.Input.(*RerankInput); ok && in != nil {
			input = *in
		}
	}
	if input.TopK <= 0 || input.TopK > len(input.Docs) {
		input.TopK = len(input.Docs)
	}

	docs, scores, stats, err := t.Reranker.Rerank(input.Query, input.Docs, input.TopK)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("reranker error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &RerankResult{
			Docs:   docs,
			Scores: scores,
			Stats:  stats,
		},
	}
}

type RAGState struct {
	Step    string
	Query   string
	Context string
}

// RangeFilter supports numeric or RFC3339 time comparisons.
type RangeFilter struct {
	Key     string   `json:"key"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	TimeMin string   `json:"time_min,omitempty"` // RFC3339
	TimeMax string   `json:"time_max,omitempty"` // RFC3339
}

func NewRAGAgent(llmToolID core.ToolID, retrievalToolID core.ToolID, rerankToolID core.ToolID) *core.FuncAgent {
	return core.NewFuncAgent("rag_orchestrator", func(ctx *core.AgentContext, msg *core.Message) {
		state, _ := ctx.UserState.(*RAGState)
		if state == nil {
			state = &RAGState{Step: "start"}
		}

		switch msg.Type {
		case core.MsgUserInput:
			state.Query = msg.Text
			state.Step = "retrieving"
			ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
				ToolID: retrievalToolID,
				Input:  &RetrievalInput{Query: msg.Text},
			})

		case core.MsgToolResult:
			switch msg.ToolRes.ToolID {
			case retrievalToolID:
				state.Step = "reranking"
				var docs []string
				switch out := msg.ToolRes.Output.(type) {
				case *RetrievalResult:
					docs = out.Docs
				case RetrievalResult:
					docs = out.Docs
				case map[string]any:
					if rawDocs, ok := out["docs"].([]string); ok {
						docs = rawDocs
					}
				}
				if len(docs) == 0 {
					docs = []string{"no docs retrieved"}
				}
				ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
					ToolID: rerankToolID,
					Input: &RerankInput{
						Query: state.Query,
						Docs:  docs,
						TopK:  3,
					},
				})

			case rerankToolID:
				state.Step = "generating"
				var docs []string
				switch out := msg.ToolRes.Output.(type) {
				case *RerankResult:
					docs = out.Docs
				case RerankResult:
					docs = out.Docs
				case map[string]any:
					if rawDocs, ok := out["docs"].([]string); ok {
						docs = rawDocs
					}
				}
				state.Context = strings.Join(docs, "\n---\n")

				systemPrompt := "You are a high-performance technical assistant."
				userPrompt := fmt.Sprintf("Question: %s\n\nContext:\n%s", state.Query, state.Context)

				ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
					ToolID: llmToolID,
					Input: &tools.LLMRequest{
						System: systemPrompt,
						Messages: []tools.LLMMessage{
							{Role: "user", Content: userPrompt},
						},
					},
				})

			case llmToolID:
				state.Step = "done"
				ctx.Scheduler.ConvMgr().MarkComplete(ctx.ConvID)
			}
		}
		ctx.UserState = state
	})
}

// ======================================================================================
// Embedder/Reranker interfaces and hash fallback
// ======================================================================================

type Embedder interface {
	Embed(text string) ([]float32, error)
	Dim() int
}

type Reranker interface {
	Rerank(query string, docs []string, topK int) ([]string, []float32, string, error)
}

type HashEmbedder struct {
	dim int
}

func NewHashEmbedder(dim int) *HashEmbedder {
	return &HashEmbedder{dim: dim}
}

func (e *HashEmbedder) Dim() int { return e.dim }

func (e *HashEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	seed := int64(h.Sum64())
	vec := make([]float32, e.dim)
	r := rand.New(rand.NewSource(seed))
	var norm float64
	for i := 0; i < e.dim; i++ {
		val := r.Float64()*2 - 1
		vec[i] = float32(val)
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vec, nil
	}
	for i := range vec {
		vec[i] /= float32(norm)
	}
	return vec, nil
}

type SimpleReranker struct {
	Embedder Embedder
}

func (r *SimpleReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	qVec, err := r.Embedder.Embed(query)
	if err != nil {
		return nil, nil, "", err
	}
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	bestDocs := make([]string, 0, topK)
	bestScores := make([]float32, 0, topK)

	for _, doc := range docs {
		dVec, err := r.Embedder.Embed(doc)
		if err != nil {
			continue
		}
		score := DotProduct(qVec, dVec)
		if len(bestDocs) < topK {
			bestDocs = append(bestDocs, doc)
			bestScores = append(bestScores, score)
			continue
		}
		minIdx := 0
		for i := 1; i < len(bestScores); i++ {
			if bestScores[i] < bestScores[minIdx] {
				minIdx = i
			}
		}
		if score > bestScores[minIdx] {
			bestScores[minIdx] = score
			bestDocs[minIdx] = doc
		}
	}
	return bestDocs, bestScores, "Simple rerank", nil
}

// ======================================================================================
// Utility
// ======================================================================================

func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func hashID(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func matchesMeta(meta map[string]string, filt map[string]string) bool {
	if len(filt) == 0 {
		return true
	}
	for k, v := range filt {
		if mv, ok := meta[k]; !ok || mv != v {
			return false
		}
	}
	return true
}

func matchesAny(meta map[string]string, any []map[string]string) bool {
	for _, f := range any {
		if matchesMeta(meta, f) {
			return true
		}
	}
	return false
}

func matchesRanges(meta map[string]string, num map[string]float64, times map[string]time.Time, ranges []RangeFilter) bool {
	if len(ranges) == 0 {
		return true
	}
	for _, rf := range ranges {
		val, ok := meta[rf.Key]
		if !ok {
			return false
		}
		// Try time bounds if provided.
		if rf.TimeMin != "" || rf.TimeMax != "" {
			mv, okT := times[rf.Key]
			if !okT {
				var err error
				mv, err = time.Parse(time.RFC3339, val)
				if err != nil {
					return false
				}
			}
			if rf.TimeMin != "" {
				minT, err := time.Parse(time.RFC3339, rf.TimeMin)
				if err != nil || mv.Before(minT) {
					return false
				}
			}
			if rf.TimeMax != "" {
				maxT, err := time.Parse(time.RFC3339, rf.TimeMax)
				if err != nil || mv.After(maxT) {
					return false
				}
			}
			continue
		}
		// Numeric bounds if set.
		if rf.Min != nil || rf.Max != nil {
			fv, okN := num[rf.Key]
			if !okN {
				var err error
				fv, err = strconv.ParseFloat(val, 64)
				if err != nil {
					return false
				}
			}
			if rf.Min != nil && fv < *rf.Min {
				return false
			}
			if rf.Max != nil && fv > *rf.Max {
				return false
			}
		}
	}
	return true
}

// tokenize is a simple Unicode-aware tokenizer used for metadata/text analysis.
func tokenize(text string) []string {
	tokens := make([]string, 0, len(text)/4+1)
	var buf strings.Builder

	flush := func() {
		if buf.Len() == 0 {
			return
		}
		tok := strings.ToLower(strings.TrimFunc(buf.String(), func(r rune) bool {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		}))
		buf.Reset()
		if tok != "" {
			tokens = append(tokens, tok)
		}
	}

	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			flush()
		default:
			buf.WriteRune(r)
		}
	}
	flush()
	return tokens
}

type walEntry struct {
	Op   string
	ID   string
	Doc  string
	Meta map[string]string
	Vec  []float32
	Coll string
}

func (vs *VectorStore) appendWAL(op, id, doc string, meta map[string]string, vec []float32) {
	if vs.walPath == "" {
		return
	}
	vs.walMu.Lock()
	defer vs.walMu.Unlock()
	f, err := os.OpenFile(vs.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer f.Close()

	entry := walEntry{Op: op, ID: id, Doc: doc, Meta: meta, Vec: vec, Coll: func() string {
		if id == "" {
			return ""
		}
		return vs.Coll[hashID(id)]
	}()}
	if err := json.NewEncoder(f).Encode(&entry); err != nil {
		return
	}
	vs.walOps++

	var doSnapshot bool
	if vs.walMaxOps > 0 && vs.walOps >= vs.walMaxOps {
		doSnapshot = true
		vs.walOps = 0
	}
	if !doSnapshot && vs.walMaxBytes > 0 {
		if info, err := f.Stat(); err == nil && info.Size() >= vs.walMaxBytes {
			doSnapshot = true
			vs.walOps = 0
		}
	}
	if doSnapshot {
		snapPath := strings.TrimSuffix(vs.walPath, ".wal")
		go func() {
			if err := vs.Save(snapPath); err == nil {
				_ = os.Truncate(vs.walPath, 0)
			}
		}()
	}
}

func replayWAL(vs *VectorStore) {
	if vs.walPath == "" {
		return
	}
	f, err := os.Open(vs.walPath)
	if err != nil {
		return
	}
	defer f.Close()

	// Disable WAL writes during replay to avoid re-appending the same ops.
	path := vs.walPath
	vs.walPath = ""
	defer func() { vs.walPath = path }()

	dec := json.NewDecoder(f)
	for {
		var e walEntry
		if err := dec.Decode(&e); err != nil {
			break
		}
		switch e.Op {
		case "insert":
			vs.Add(e.Vec, e.Doc, e.ID, e.Meta, e.Coll)
		case "upsert":
			vs.Upsert(e.Vec, e.Doc, e.ID, e.Meta, e.Coll)
		case "delete":
			vs.Delete(e.ID)
		}
	}
	_ = os.Remove(path)
}

func (vs *VectorStore) computeChecksum() string {
	return fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d-%d", vs.Count, vs.next, len(vs.Docs))))
}

func (vs *VectorStore) validateChecksum() bool {
	if vs.checksum == "" {
		return true
	}
	return vs.checksum == vs.computeChecksum()
}

// ======================================================================================
// Embedder selection (Hash by default; ONNX under build tag)
// ======================================================================================

func initEmbedder(defaultDim int) Embedder {
	defaultModel := "vectordb/models/bge-small-en-v1.5/model.onnx"
	defaultTok := "vectordb/models/bge-small-en-v1.5/tokenizer.json"

	modelPath := os.Getenv("ONNX_EMBED_MODEL")
	tokPath := os.Getenv("ONNX_EMBED_TOKENIZER")
	if modelPath == "" {
		if _, err := os.Stat(defaultModel); err == nil {
			modelPath = defaultModel
		}
	}
	if tokPath == "" {
		if _, err := os.Stat(defaultTok); err == nil {
			tokPath = defaultTok
		}
	}
	maxLen := 512
	if env := os.Getenv("ONNX_EMBED_MAX_LEN"); env != "" {
		if v, err := strconv.Atoi(env); err == nil && v > 0 {
			maxLen = v
		}
	}
	if modelPath != "" && tokPath != "" {
		if emb, err := NewOnnxEmbedder(modelPath, tokPath, defaultDim, maxLen); err == nil {
			fmt.Println("Using ONNX embedder:", modelPath)
			return emb
		}
		fmt.Println("Falling back to hash embedder (ONNX init failed)")
	}
	return NewHashEmbedder(defaultDim)
}

func initReranker(embedder Embedder) Reranker {
	modelPath := os.Getenv("ONNX_RERANK_MODEL")
	tokPath := os.Getenv("ONNX_RERANK_TOKENIZER")
	maxLen := 512
	if env := os.Getenv("ONNX_RERANK_MAX_LEN"); env != "" {
		if v, err := strconv.Atoi(env); err == nil && v > 0 {
			maxLen = v
		}
	}
	if modelPath != "" && tokPath != "" {
		if rr, err := NewOnnxCrossEncoderReranker(modelPath, tokPath, maxLen); err == nil {
			fmt.Println("Using ONNX reranker:", modelPath)
			return rr
		}
		fmt.Println("Falling back to simple reranker (ONNX init failed)")
	}
	return &SimpleReranker{Embedder: embedder}
}

// ======================================================================================
// Main: bootstrap, HTTP API
// ======================================================================================

func main() {
	fmt.Println(">>> Initializing Flat Buffer Vector Engine...")
	const indexPath = "vectordb/index.gob"
	defaultDim := 384
	if envDim := os.Getenv("EMBED_DIM"); envDim != "" {
		if v, err := strconv.Atoi(envDim); err == nil && v > 0 {
			defaultDim = v
		}
	}

	initMetrics()

	embedder := initEmbedder(defaultDim)
	store, loaded := loadOrInitStore(indexPath, 100000, embedder.Dim())
	store.walMaxBytes = envInt64("WAL_MAX_BYTES", 5*1024*1024)
	store.walMaxOps = envInt("WAL_MAX_OPS", 1000)

	if !loaded {
		fmt.Println(">>> Hydrating Index with 100k vectors...")
		for i := 0; i < 100000; i++ {
			vec, _ := embedder.Embed(fmt.Sprintf("doc-%d", i))
			store.Add(vec, fmt.Sprintf("doc-%d content about memory locality and vector search.", i), "", nil, "default")
		}
		if err := store.Save(indexPath); err != nil {
			fmt.Printf("warning: failed to save index: %v\n", err)
		}
	}
	fmt.Printf(">>> Index Ready. %d Vectors in Contiguous RAM.\n", store.Count)

	sched := core.NewScheduler(core.DefaultSchedulerConfig)
	defer sched.Shutdown()
	go sched.Run()

	retrievalTool := &RetrievalTool{Store: store, Embedder: embedder}
	retrievalID := sched.Tools().Register(retrievalTool, core.ToolPolicy{
		MaxRetries:     0,
		DefaultTimeout: 200 * time.Millisecond,
	}, nil)

	reranker := initReranker(embedder)
	rerankTool := &RerankTool{Reranker: reranker}
	rerankID := sched.Tools().Register(rerankTool, core.ToolPolicy{
		MaxRetries:     0,
		DefaultTimeout: 200 * time.Millisecond,
	}, nil)

	router := tools.NewModelRouter()
	llmCfg, err := router.FastestModel()
	if err != nil {
		fmt.Println("Warning: No API key found, LLM calls may fail.")
	}
	llmID := sched.Tools().Register(tools.NewLLMTool(llmCfg), core.DefaultToolPolicy, nil)

	agent := NewRAGAgent(llmID, retrievalID, rerankID)
	agentID := sched.Agents().Register(agent, nil)

	// HTTP API
	go func() {
		handler := newHTTPHandler(store, embedder, reranker, indexPath)
		addr := ":8080"
		fmt.Printf(">>> HTTP API listening on %s (POST /insert, POST /batch_insert, POST /query, POST /delete, GET /health, GET /metrics)\n", addr)
		_ = http.ListenAndServe(addr, handler)
	}()

	fmt.Println(">>> Starting RAG Conversation...")
	convID := sched.StartConversation(agentID, "Why is memory locality important for vector search?")

	ticker := time.NewTicker(500 * time.Millisecond)
	timeout := time.After(10 * time.Second)
	for {
		select {
		case <-timeout:
			fmt.Println("Timeout reached (Demo end)")
			return
		case <-ticker.C:
			if conv, ok := sched.ConvMgr().Get(convID); ok && conv.State == core.ConvComplete {
				fmt.Println(">>> Conversation Marked Complete by Agent.")
				return
			}
		}
	}
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func envInt64(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func fileSize(path string) int64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
