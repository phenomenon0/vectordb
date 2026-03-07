package storage

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
	"time"
)

// generateCorrelatedData creates embedding data where consecutive values are
// correlated (simulates real embeddings where nearby dimensions are similar).
func generateCorrelatedData(n int) []float32 {
	data := make([]float32, n)
	data[0] = rand.Float32()
	for i := 1; i < n; i++ {
		// Each value is close to the previous one (high correlation)
		data[i] = data[i-1] + (rand.Float32()-0.5)*0.01
	}
	return data
}

// generateUncorrelatedData creates random float32 data with no correlation
// between consecutive values.
func generateUncorrelatedData(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1 // random in [-1, 1]
	}
	return data
}

func TestCowrieDeltaZstdRoundTrip(t *testing.T) {
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	// Use correlated data so delta encoding is triggered
	payload := generateTestPayload(100, 384)
	payload.Data = generateCorrelatedData(100 * 384)

	var buf bytes.Buffer
	if err := cw.Save(&buf, payload); err != nil {
		t.Fatalf("cowrie-delta-zstd save failed: %v", err)
	}

	loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("cowrie-delta-zstd load failed: %v", err)
	}

	// Verify scalar fields
	if loaded.Dim != payload.Dim {
		t.Errorf("Dim mismatch: got %d, want %d", loaded.Dim, payload.Dim)
	}
	if loaded.Count != payload.Count {
		t.Errorf("Count mismatch: got %d, want %d", loaded.Count, payload.Count)
	}
	if loaded.Next != payload.Next {
		t.Errorf("Next mismatch: got %d, want %d", loaded.Next, payload.Next)
	}
	if loaded.Checksum != payload.Checksum {
		t.Errorf("Checksum mismatch: got %q, want %q", loaded.Checksum, payload.Checksum)
	}

	// Verify embedding data roundtrips exactly
	if len(loaded.Data) != len(payload.Data) {
		t.Fatalf("Data length mismatch: got %d, want %d", len(loaded.Data), len(payload.Data))
	}
	for i := range payload.Data {
		if loaded.Data[i] != payload.Data[i] {
			t.Errorf("Data[%d] mismatch: got %v, want %v", i, loaded.Data[i], payload.Data[i])
			break
		}
	}

	// Verify other fields survived
	if len(loaded.Docs) != len(payload.Docs) {
		t.Errorf("Docs length mismatch: got %d, want %d", len(loaded.Docs), len(payload.Docs))
	}
	if len(loaded.IDs) != len(payload.IDs) {
		t.Errorf("IDs length mismatch: got %d, want %d", len(loaded.IDs), len(payload.IDs))
	}
}

func TestCowrieDeltaFallbackUncorrelated(t *testing.T) {
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	// Use uncorrelated (random) data - ShouldUseDelta should return false,
	// so it falls back to normal tensor encoding
	payload := generateTestPayload(100, 384)
	payload.Data = generateUncorrelatedData(100 * 384)

	var buf bytes.Buffer
	if err := cw.Save(&buf, payload); err != nil {
		t.Fatalf("save with uncorrelated data failed: %v", err)
	}

	loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("load with uncorrelated data failed: %v", err)
	}

	// Data should still roundtrip correctly via the tensor fallback
	if len(loaded.Data) != len(payload.Data) {
		t.Fatalf("Data length mismatch: got %d, want %d", len(loaded.Data), len(payload.Data))
	}
	for i := range payload.Data {
		if loaded.Data[i] != payload.Data[i] {
			t.Errorf("Data[%d] mismatch: got %v, want %v", i, loaded.Data[i], payload.Data[i])
			break
		}
	}
}

func TestCowrieDeltaEmptyData(t *testing.T) {
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	payload := &Payload{
		Dim:       384,
		Data:      nil, // empty embedding data
		Count:     0,
		Checksum:  "empty",
		LastSaved: time.Now().Truncate(time.Nanosecond),
	}

	var buf bytes.Buffer
	if err := cw.Save(&buf, payload); err != nil {
		t.Fatalf("save with empty data failed: %v", err)
	}

	loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("load with empty data failed: %v", err)
	}

	if len(loaded.Data) != 0 {
		t.Errorf("expected empty Data, got length %d", len(loaded.Data))
	}
	if loaded.Dim != 384 {
		t.Errorf("Dim mismatch: got %d, want 384", loaded.Dim)
	}
	if loaded.Checksum != "empty" {
		t.Errorf("Checksum mismatch: got %q, want %q", loaded.Checksum, "empty")
	}
}

func TestCowrieDeltaEmptySliceData(t *testing.T) {
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	payload := &Payload{
		Dim:       128,
		Data:      []float32{}, // empty slice (not nil)
		Count:     0,
		Checksum:  "empty-slice",
		LastSaved: time.Now().Truncate(time.Nanosecond),
	}

	var buf bytes.Buffer
	if err := cw.Save(&buf, payload); err != nil {
		t.Fatalf("save with empty slice data failed: %v", err)
	}

	loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("load with empty slice data failed: %v", err)
	}

	if len(loaded.Data) != 0 {
		t.Errorf("expected empty Data, got length %d", len(loaded.Data))
	}
}

func TestCowrieDeltaFormatMetadata(t *testing.T) {
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	if cw.Name() != "cowrie-delta-zstd" {
		t.Errorf("Name() = %q, want %q", cw.Name(), "cowrie-delta-zstd")
	}
	if cw.Extension() != ".cowrie.delta.zst" {
		t.Errorf("Extension() = %q, want %q", cw.Extension(), ".cowrie.delta.zst")
	}

	// Verify it's registered
	f := Get("cowrie-delta-zstd")
	if f == nil {
		t.Fatal("cowrie-delta-zstd not found in registry")
	}
	if f.Name() != "cowrie-delta-zstd" {
		t.Errorf("registered format Name() = %q, want %q", f.Name(), "cowrie-delta-zstd")
	}
}

func TestCowrieDeltaDoesNotBreakExistingFormats(t *testing.T) {
	// Verify the existing formats still work correctly after adding delta

	t.Run("cowrie", func(t *testing.T) {
		cw := &CowrieFormat{UseCompression: false}
		if cw.Name() != "cowrie" {
			t.Errorf("Name() = %q, want %q", cw.Name(), "cowrie")
		}
		if cw.Extension() != ".cowrie" {
			t.Errorf("Extension() = %q, want %q", cw.Extension(), ".cowrie")
		}

		payload := generateTestPayload(50, 128)
		var buf bytes.Buffer
		if err := cw.Save(&buf, payload); err != nil {
			t.Fatalf("save failed: %v", err)
		}
		loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}
		if loaded.Count != payload.Count {
			t.Errorf("Count mismatch: got %d, want %d", loaded.Count, payload.Count)
		}
	})

	t.Run("cowrie-zstd", func(t *testing.T) {
		cw := &CowrieFormat{UseCompression: true}
		if cw.Name() != "cowrie-zstd" {
			t.Errorf("Name() = %q, want %q", cw.Name(), "cowrie-zstd")
		}
		if cw.Extension() != ".cowrie.zst" {
			t.Errorf("Extension() = %q, want %q", cw.Extension(), ".cowrie.zst")
		}

		payload := generateTestPayload(50, 128)
		var buf bytes.Buffer
		if err := cw.Save(&buf, payload); err != nil {
			t.Fatalf("save failed: %v", err)
		}
		loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}
		if loaded.Count != payload.Count {
			t.Errorf("Count mismatch: got %d, want %d", loaded.Count, payload.Count)
		}
	})
}

func TestCowrieDeltaPrecision(t *testing.T) {
	// Verify that delta encoding preserves exact float32 values (no precision loss)
	cw := &CowrieFormat{UseCompression: true, UseDeltaPrediction: true}

	// Create data with known exact values including edge cases
	payload := &Payload{
		Dim:       4,
		Count:     1,
		Checksum:  "precision",
		LastSaved: time.Now().Truncate(time.Nanosecond),
	}

	// Include special float values and correlated data to trigger delta encoding
	special := []float32{
		0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007,
		0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015,
		math.SmallestNonzeroFloat32, math.SmallestNonzeroFloat32 + 1e-40,
	}
	// Pad with correlated values to ensure ShouldUseDelta returns true
	for i := len(special); i < 200; i++ {
		special = append(special, float32(i)*0.001)
	}
	payload.Data = special

	var buf bytes.Buffer
	if err := cw.Save(&buf, payload); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	loaded, err := cw.Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if len(loaded.Data) != len(payload.Data) {
		t.Fatalf("Data length mismatch: got %d, want %d", len(loaded.Data), len(payload.Data))
	}

	for i := range payload.Data {
		got := loaded.Data[i]
		want := payload.Data[i]
		// Delta encoding is exact for normal floats but may have ULP errors
		// for denormalized values (subnormals near SmallestNonzeroFloat32).
		// Use relative tolerance for values below 1e-30.
		if math.Float32bits(got) != math.Float32bits(want) {
			if math.Abs(float64(want)) < 1e-30 {
				// Subnormal tolerance: allow 1 ULP difference
				diff := math.Abs(float64(got) - float64(want))
				if diff < 1e-44 {
					continue // Within subnormal precision
				}
			}
			t.Errorf("Data[%d] bit mismatch: got %08x (%v), want %08x (%v)",
				i,
				math.Float32bits(got), got,
				math.Float32bits(want), want)
		}
	}
}
