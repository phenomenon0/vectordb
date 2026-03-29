//go:build onnx

package main

import (
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"strings"
	"sync"
	"unicode"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	padID int64 = 0
	clsID int64 = 1
	sepID int64 = 2
	unkID int64 = 3

	hashBaseID int64  = 10
	hashMod    uint64 = 200_000
)

// hashToken maps a token string to a stable int64 ID.
func hashToken(s string) int64 {
	switch s {
	case "[CLS]":
		return clsID
	case "[SEP]":
		return sepID
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return hashBaseID + int64(h.Sum64()%hashMod)
}

// normalizeToken lowers case and trims punctuation/symbols at both ends.
func normalizeToken(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSymbol(r)
	})
	return s
}

func isSentencePunct(r rune) bool {
	switch r {
	case '.', '!', '?', ',', ';', ':':
		return true
	default:
		return false
	}
}

// onnxTokenize splits on Unicode whitespace, peels punctuation into its own tokens, normalizes.
func onnxTokenize(text string) []string {
	tokens := make([]string, 0, len(text)/4+1)
	var buf strings.Builder

	flush := func() {
		if buf.Len() == 0 {
			return
		}
		tok := normalizeToken(buf.String())
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
			if isSentencePunct(r) {
				tokens = append(tokens, string(r))
			}
		default:
			buf.WriteRune(r)
		}
	}
	flush()
	return tokens
}

// charNgrams yields character n-grams for robustness.
func charNgrams(tok string, minN, maxN int) []string {
	runes := []rune(tok)
	n := len(runes)
	out := make([]string, 0, n)
	for l := minN; l <= maxN; l++ {
		if l > n {
			break
		}
		for i := 0; i+l <= n; i++ {
			out = append(out, "ng:"+string(runes[i:i+l]))
		}
	}
	return out
}

// BertTokenizer wraps sugarme tokenizer; hashes OOVs and adds char n-grams.
type BertTokenizer struct {
	tok       *tokenizer.Tokenizer
	maxTokens int
}

func LoadBertTokenizer(tokenizerPath string, maxTokens int) (*BertTokenizer, error) {
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	return &BertTokenizer{tok: tk, maxTokens: maxTokens}, nil
}

func (t *BertTokenizer) Encode(text string) ([]int64, []int64) {
	if t.tok == nil {
		return nil, nil
	}
	enc, err := t.tok.EncodeSingle(text, true)
	if err != nil {
		return nil, nil
	}
	idsRaw := enc.GetIds()
	maskRaw := enc.GetAttentionMask()
	ids := make([]int64, t.maxTokens)
	mask := make([]int64, t.maxTokens)
	n := len(idsRaw)
	if n > t.maxTokens {
		n = t.maxTokens
	}
	for i := 0; i < n; i++ {
		ids[i] = int64(idsRaw[i])
		mask[i] = int64(maskRaw[i])
	}
	return ids, mask

	// Fallback path removed for reranker strictness.

}

func (t *BertTokenizer) EncodePair(query, doc string) ([]int64, []int64) {
	if t.tok == nil {
		return nil, nil
	}
	enc, err := t.tok.EncodePair(query, doc, true)
	if err != nil {
		return nil, nil
	}
	idsRaw := enc.GetIds()
	maskRaw := enc.GetAttentionMask()
	ids := make([]int64, t.maxTokens)
	mask := make([]int64, t.maxTokens)
	n := len(idsRaw)
	if n > t.maxTokens {
		n = t.maxTokens
	}
	for i := 0; i < n; i++ {
		ids[i] = int64(idsRaw[i])
		mask[i] = int64(maskRaw[i])
	}
	return ids, mask
}

// ===== Embedder =====

type OnnxEmbedder struct {
	sess     *ort.DynamicSession[int64, float32]
	maxLen   int
	dim      int
	fallback *HashEmbedder
	tok      *BertTokenizer
}

var ortOnce sync.Once

func initOrt() error {
	var initErr error
	ortOnce.Do(func() {
		if lib := os.Getenv("ONNX_SHARED_LIB"); lib != "" {
			ort.SetSharedLibraryPath(lib)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

func NewOnnxEmbedder(modelPath, tokenizerPath string, dim, maxLen int) (Embedder, error) {
	if err := initOrt(); err != nil {
		return nil, fmt.Errorf("init ort env: %w", err)
	}
	tok, err := LoadBertTokenizer(tokenizerPath, maxLen)
	if err != nil {
		return nil, err
	}
	sess, err := ort.NewDynamicSession[int64, float32](modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"})
	if err != nil {
		return nil, fmt.Errorf("new session: %w", err)
	}

	return &OnnxEmbedder{
		sess:     sess,
		maxLen:   maxLen,
		dim:      dim,
		fallback: NewHashEmbedder(dim),
		tok:      tok,
	}, nil
}

func (o *OnnxEmbedder) Dim() int { return o.dim }

func (o *OnnxEmbedder) EmbedQuery(text string) ([]float32, error) { return o.Embed(text) }

func (o *OnnxEmbedder) Embed(text string) ([]float32, error) {
	if o.sess == nil || o.tok == nil {
		return o.fallback.Embed(text)
	}

	ids, mask := o.tok.Encode(text)

	inputIDs, err := ort.NewTensor(ort.NewShape(1, int64(o.maxLen)), ids)
	if err != nil {
		return o.fallback.Embed(text)
	}
	defer inputIDs.Destroy()

	attnMask, err := ort.NewTensor(ort.NewShape(1, int64(o.maxLen)), mask)
	if err != nil {
		return o.fallback.Embed(text)
	}
	defer attnMask.Destroy()

	// Expect last_hidden_state shape [1, maxLen, dim]
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(o.maxLen), int64(o.dim)))
	if err != nil {
		return o.fallback.Embed(text)
	}
	defer outputTensor.Destroy()

	if err := o.sess.Run([]*ort.Tensor[int64]{inputIDs, attnMask}, []*ort.Tensor[float32]{outputTensor}); err != nil {
		return o.fallback.Embed(text)
	}

	vec := outputTensor.GetData()
	if len(vec) == 0 {
		return o.fallback.Embed(text)
	}

	// Mean-pool using attention mask count.
	tokenCount := 0
	for _, v := range mask {
		if v == 1 {
			tokenCount++
		}
	}
	if tokenCount <= 0 {
		tokenCount = 1
	}

	pooled := make([]float32, o.dim)
	seq := len(vec) / o.dim
	if seq <= 0 {
		return o.fallback.Embed(text)
	}
	for i := 0; i < seq; i++ {
		base := i * o.dim
		for j := 0; j < o.dim; j++ {
			pooled[j] += vec[base+j]
		}
	}
	inv := float32(1.0 / float64(tokenCount))
	for j := 0; j < o.dim; j++ {
		pooled[j] *= inv
	}

	// L2 normalize
	var norm float64
	for _, v := range pooled {
		norm += float64(v * v)
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range pooled {
			pooled[i] /= float32(norm)
		}
	}
	return pooled, nil
}

// ===== Reranker =====

type OnnxCrossEncoderReranker struct {
	sess     *ort.DynamicSession[int64, float32]
	maxLen   int
	fallback Reranker
	tok      *BertTokenizer
}

func NewOnnxCrossEncoderReranker(modelPath, tokenizerPath string, maxLen int) (Reranker, error) {
	if err := initOrt(); err != nil {
		return nil, fmt.Errorf("init ort env: %w", err)
	}
	tok, err := LoadBertTokenizer(tokenizerPath, maxLen)
	if err != nil {
		return nil, err
	}
	// Adjust input/output names if your model differs.
	sess, err := ort.NewDynamicSession[int64, float32](modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"})
	if err != nil {
		return nil, fmt.Errorf("new rerank session: %w", err)
	}
	return &OnnxCrossEncoderReranker{
		sess:     sess,
		maxLen:   maxLen,
		fallback: &SimpleReranker{Embedder: NewHashEmbedder(384)},
		tok:      tok,
	}, nil
}

func (r *OnnxCrossEncoderReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	type scored struct {
		doc   string
		score float32
	}
	results := make([]scored, 0, len(docs))

	for _, doc := range docs {
		ids, mask := r.tok.EncodePair(query, doc)
		if ids == nil || mask == nil {
			return r.fallback.Rerank(query, docs, topK)
		}

		inputIDs, err := ort.NewTensor(ort.NewShape(1, int64(r.maxLen)), ids)
		if err != nil {
			return r.fallback.Rerank(query, docs, topK)
		}
		attnMask, err := ort.NewTensor(ort.NewShape(1, int64(r.maxLen)), mask)
		if err != nil {
			inputIDs.Destroy()
			return r.fallback.Rerank(query, docs, topK)
		}
		output := make([]float32, 1)
		outTensor, err := ort.NewTensor(ort.NewShape(1, 1), output)
		if err != nil {
			inputIDs.Destroy()
			attnMask.Destroy()
			return r.fallback.Rerank(query, docs, topK)
		}

		err = r.sess.Run([]*ort.Tensor[int64]{inputIDs, attnMask}, []*ort.Tensor[float32]{outTensor})

		inputIDs.Destroy()
		attnMask.Destroy()
		defer outTensor.Destroy()

		if err != nil {
			return r.fallback.Rerank(query, docs, topK)
		}
		data := outTensor.GetData()
		score := float32(0)
		if len(data) > 0 {
			score = data[0]
		}
		results = append(results, scored{doc: doc, score: score})
	}

	// sort by score desc
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
	if len(results) > topK {
		results = results[:topK]
	}
	docsOut := make([]string, 0, len(results))
	scores := make([]float32, 0, len(results))
	for _, r := range results {
		docsOut = append(docsOut, r.doc)
		scores = append(scores, r.score)
	}
	return docsOut, scores, "ONNX cross-encoder rerank", nil
}
