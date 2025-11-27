//go:build !onnx
// +build !onnx

package main

import "fmt"

// OnnxEmbedder is a placeholder when built without the "onnx" tag.
type OnnxEmbedder struct{}

func (o *OnnxEmbedder) Embed(text string) ([]float32, error) {
	return nil, fmt.Errorf("onnx embedder not enabled (build with -tags onnx)")
}

func (o *OnnxEmbedder) Dim() int { return 0 }

// OnnxCrossEncoderReranker is a placeholder when built without the "onnx" tag.
type OnnxCrossEncoderReranker struct{}

func (r *OnnxCrossEncoderReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	return nil, nil, "", fmt.Errorf("onnx reranker not enabled (build with -tags onnx)")
}

func NewOnnxEmbedder(modelPath, tokenizerPath string, dim, maxLen int) (Embedder, error) {
	return nil, fmt.Errorf("onnx embedder not enabled (build with -tags onnx)")
}

func NewOnnxCrossEncoderReranker(modelPath, tokenizerPath string, maxLen int) (Reranker, error) {
	return nil, fmt.Errorf("onnx reranker not enabled (build with -tags onnx)")
}
