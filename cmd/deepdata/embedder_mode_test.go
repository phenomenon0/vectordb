package main

import "testing"

func TestLocalModeDoesNotAutoSelectOpenAI(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("EMBEDDER_TYPE", "")
	t.Setenv("ONNX_EMBED_MODEL", "/missing/model.onnx")
	t.Setenv("ONNX_EMBED_TOKENIZER", "/missing/tokenizer.json")
	t.Setenv("OLLAMA_URL", "http://127.0.0.1:1")

	mode := &ModeConfig{Mode: ModeLocal}
	emb, err := NewEmbedderFactory(mode, nil).CreateEmbedder()
	if err != nil {
		t.Fatalf("create embedder failed: %v", err)
	}
	if emb == nil {
		t.Fatal("expected embedder instance")
	}
	if mode.EmbedderType == "openai" {
		t.Fatalf("expected local mode to avoid implicit openai selection")
	}
}

func TestLocalModeCanExplicitlySelectOpenAI(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("EMBEDDER_TYPE", "openai")

	mode := &ModeConfig{Mode: ModeLocal}
	emb, err := NewEmbedderFactory(mode, nil).CreateEmbedder()
	if err != nil {
		t.Fatalf("create embedder failed: %v", err)
	}
	if emb == nil {
		t.Fatal("expected embedder instance")
	}
	if mode.EmbedderType != "openai" {
		t.Fatalf("expected explicit openai selection, got %q", mode.EmbedderType)
	}
	if mode.Dimension != 1536 {
		t.Fatalf("expected openai dimension 1536, got %d", mode.Dimension)
	}
}
