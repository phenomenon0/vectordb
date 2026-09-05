package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"

	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// serverEmbedder is the one text embedder a process runs, named by
// DEEPDATA_EMBEDDER. A nil *serverEmbedder means none: `texts` are refused
// with embedder_unavailable and callers send vectors. It sits above the
// engine, which stays vectors-only.
type serverEmbedder struct {
	Embedder
	Provider string
	Model    string
}

// Label is the provider:model string reported by /readyz and embedded_by.
func (e *serverEmbedder) Label() string {
	if e == nil {
		return "none"
	}
	return e.Provider + ":" + e.Model
}

func (e *serverEmbedder) matches(cfg *vcollection.EmbeddingConfig) bool {
	return e != nil && cfg.Provider == e.Provider && (cfg.Model == "" || cfg.Model == e.Model)
}

// newServerEmbedder selects the process embedder. none is the default; hash
// is never implicit. Provider-backed embedders are probed once so a
// configured-but-unreachable embedder refuses startup instead of failing the
// first request.
func newServerEmbedder(cfg embedderConfig) (*serverEmbedder, error) {
	switch cfg.Kind {
	case "", "none":
		return nil, nil
	case "hash":
		dim := cfg.Dim
		if dim <= 0 || dim > vcollection.MaxVectorDimension {
			return nil, fmt.Errorf("DEEPDATA_EMBED_DIM must be in 1..%d, got %d", vcollection.MaxVectorDimension, dim)
		}
		return &serverEmbedder{Embedder: NewHashEmbedder(dim), Provider: "hash", Model: strconv.Itoa(dim)}, nil
	case "ollama":
		baseURL := cfg.OllamaURL
		if baseURL == "" {
			baseURL = "http://localhost:11434"
		}
		model := cfg.OllamaModel
		if model == "" {
			model = "nomic-embed-text"
		}
		emb := NewOllamaEmbedder(baseURL, model)
		vec, err := emb.Embed("deepdata startup probe")
		if err != nil {
			return nil, fmt.Errorf("ollama model %s at %s: %w", model, baseURL, err)
		}
		emb.dim = len(vec)
		return &serverEmbedder{Embedder: emb, Provider: "ollama", Model: model}, nil
	case "openai":
		if cfg.OpenAIAPIKey == "" {
			return nil, errors.New("DEEPDATA_EMBEDDER=openai requires OPENAI_API_KEY")
		}
		emb := NewOpenAIEmbedder(cfg.OpenAIAPIKey)
		if _, err := emb.Embed("deepdata startup probe"); err != nil {
			return nil, fmt.Errorf("openai model %s: %w", emb.model, err)
		}
		return &serverEmbedder{Embedder: emb, Provider: "openai", Model: emb.model}, nil
	case "onnx":
		modelPath := cfg.OnnxModel
		if modelPath == "" {
			modelPath = "vectordb/models/bge-small-en-v1.5/model.onnx"
		}
		tokPath := cfg.OnnxTokenizer
		if tokPath == "" {
			tokPath = "vectordb/models/bge-small-en-v1.5/tokenizer.json"
		}
		emb, err := NewOnnxEmbedder(modelPath, tokPath, cfg.Dim, cfg.OnnxMaxLen)
		if err != nil {
			return nil, fmt.Errorf("onnx model %s: %w", modelPath, err)
		}
		if _, err := emb.Embed("deepdata startup probe"); err != nil {
			return nil, fmt.Errorf("onnx model %s: %w", modelPath, err)
		}
		return &serverEmbedder{Embedder: emb, Provider: "onnx", Model: filepath.Base(filepath.Dir(modelPath))}, nil
	}
	return nil, fmt.Errorf("unknown DEEPDATA_EMBEDDER %q (none|ollama|openai|onnx|hash)", cfg.Kind)
}

// resolveSchemaEmbedding checks every dense embedding binding against the
// process embedder before the schema is journaled: fills dim (when 0) and
// model (when empty) from the embedder, rejects a provider/model or dim the
// server cannot honor. Sparse bm25 bindings need no embedder; the engine
// validates them.
func resolveSchemaEmbedding(schema *vcollection.CollectionSchema, emb *serverEmbedder) *apierror.Error {
	for i := range schema.Fields {
		field := &schema.Fields[i]
		if field.Embedding == nil || field.Type == vcollection.VectorTypeSparse {
			continue
		}
		if emb == nil {
			e := apierror.New(apierror.CodeEmbedderUnavailable, fmt.Sprintf("field %s binds embedding %s but this server has no text embedder", field.Name, field.Embedding.Provider))
			e.Field = "fields." + field.Name + ".embedding"
			return e
		}
		if !emb.matches(field.Embedding) {
			e := apierror.New(apierror.CodeEmbeddingMismatch, fmt.Sprintf("field %s binds embedding %s:%s; this server embeds with %s", field.Name, field.Embedding.Provider, field.Embedding.Model, emb.Label()))
			e.Field = "fields." + field.Name + ".embedding"
			return e
		}
		field.Embedding.Model = emb.Model
		if field.Dim == 0 {
			field.Dim = emb.Dim()
		} else if field.Dim != emb.Dim() {
			e := apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("field %s dim %d does not match embedder %s dim %d; omit dim to take the embedder's", field.Name, field.Dim, emb.Label(), emb.Dim()))
			e.Field = "fields." + field.Name + ".dim"
			return e
		}
	}
	return nil
}

// resolveTexts embeds each text into vectors under the same field name and
// returns embedded_by (field → provider:model). Shared by HTTP and gRPC for
// documents (isQuery=false, Embed) and queries (isQuery=true, EmbedQuery).
func resolveTexts(fields []vcollection.FieldInfo, emb *serverEmbedder, texts map[string]string, vectors map[string]interface{}, isQuery bool) (map[string]string, *apierror.Error) {
	embeddedBy := make(map[string]string, len(texts))
	for name, text := range texts {
		fail := func(code, msg string) (map[string]string, *apierror.Error) {
			e := apierror.New(code, msg)
			e.Field = "texts." + name
			return nil, e
		}
		if _, dup := vectors[name]; dup {
			return fail(apierror.CodeInvalidArgument, fmt.Sprintf("field %s was sent as both a text and a vector", name))
		}
		var field *vcollection.FieldInfo
		for i := range fields {
			if fields[i].Name == name {
				field = &fields[i]
				break
			}
		}
		if field == nil {
			return fail(apierror.CodeInvalidArgument, fmt.Sprintf("field %s is not in the collection schema", name))
		}
		if field.Embedding == nil {
			_, e := fail(apierror.CodeInvalidArgument, fmt.Sprintf("field %s has no embedding binding", name))
			e.Hint = "bind an embedding on this field at create time or send a vector"
			return nil, e
		}
		if field.Type == vcollection.VectorTypeSparse {
			vec, err := vcollection.TextToSparse(text, field.Dim)
			if err != nil {
				return fail(apierror.CodeInvalidArgument, fmt.Sprintf("field %s: %v", name, err))
			}
			vectors[name] = vec
			embeddedBy[name] = vcollection.EmbeddingProviderBM25
			continue
		}
		if emb == nil {
			return fail(apierror.CodeEmbedderUnavailable, fmt.Sprintf("field %s needs embedder %s:%s but this server has none", name, field.Embedding.Provider, field.Embedding.Model))
		}
		if !emb.matches(field.Embedding) {
			return fail(apierror.CodeEmbeddingMismatch, fmt.Sprintf("field %s is bound to %s:%s; this server embeds with %s", name, field.Embedding.Provider, field.Embedding.Model, emb.Label()))
		}
		var vec []float32
		var err error
		if isQuery {
			vec, err = emb.EmbedQuery(text)
		} else {
			vec, err = emb.Embed(text)
		}
		if err != nil {
			return fail(apierror.CodeEmbedderUnavailable, fmt.Sprintf("embedder %s failed on field %s: %v", emb.Label(), name, err))
		}
		vectors[name] = vec
		embeddedBy[name] = emb.Label()
	}
	return embeddedBy, nil
}

// applyTexts resolves a request's texts against the collection schema; on
// failure it writes the error and returns false.
func (s *CollectionHTTPServer) applyTexts(w http.ResponseWriter, tenantID, collection string, texts map[string]string, vectors map[string]interface{}, isQuery bool) (map[string]string, bool) {
	if len(texts) == 0 {
		return nil, true
	}
	info, err := s.tenantManager.GetCollectionInfo(tenantID, collection)
	if err != nil {
		writeCanonicalOperationError(w, err, apierror.CodeNotFound)
		return nil, false
	}
	embeddedBy, aerr := resolveTexts(info.Fields, s.embedder, texts, vectors, isQuery)
	if aerr != nil {
		apierror.WriteHTTP(w, aerr)
		return nil, false
	}
	return embeddedBy, true
}

func (s *CollectionGRPCServer) applyTexts(ctx context.Context, tenantID, collection string, texts map[string]string, vectors map[string]interface{}, isQuery bool) (map[string]string, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	info, err := s.tenants.GetCollectionInfo(tenantID, collection)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
	}
	embeddedBy, aerr := resolveTexts(info.Fields, s.embedder, texts, vectors, isQuery)
	if aerr != nil {
		return nil, aerr.GRPC(ctx)
	}
	return embeddedBy, nil
}
