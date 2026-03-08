package telemetry

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func getGaugeValue(g prometheus.Gauge) float64 {
	var m dto.Metric
	g.Write(&m)
	return m.GetGauge().GetValue()
}

func getCounterValue(c prometheus.Counter) float64 {
	var m dto.Metric
	c.Write(&m)
	return m.GetCounter().GetValue()
}

func getHistogramCount(h prometheus.Observer) uint64 {
	// Observer doesn't expose Write; need the underlying Histogram metric
	if hm, ok := h.(prometheus.Metric); ok {
		var m dto.Metric
		hm.Write(&m)
		return m.GetHistogram().GetSampleCount()
	}
	return 0
}

func TestVectorsTotal(t *testing.T) {
	g := VectorsTotal.WithLabelValues("test_collection")
	g.Set(42)
	if v := getGaugeValue(g); v != 42 {
		t.Fatalf("expected 42, got %v", v)
	}
	g.Add(8)
	if v := getGaugeValue(g); v != 50 {
		t.Fatalf("expected 50, got %v", v)
	}
}

func TestSearchRequestsTotal(t *testing.T) {
	c := SearchRequestsTotal.WithLabelValues("hnsw")
	c.Inc()
	c.Inc()
	if v := getCounterValue(c); v != 2 {
		t.Fatalf("expected 2, got %v", v)
	}
}

func TestSearchDurationSeconds(t *testing.T) {
	h := SearchDurationSeconds.WithLabelValues("hnsw")
	h.Observe(0.005)
	h.Observe(0.010)
	count := getHistogramCount(h)
	if count != 2 {
		t.Fatalf("expected 2 observations, got %d", count)
	}
}

func TestInsertRequestsTotal(t *testing.T) {
	c := InsertRequestsTotal.WithLabelValues("hnsw")
	c.Inc()
	if v := getCounterValue(c); v != 1 {
		t.Fatalf("expected 1, got %v", v)
	}
}

func TestInsertDurationSeconds(t *testing.T) {
	h := InsertDurationSeconds.WithLabelValues("hnsw")
	h.Observe(0.001)
	count := getHistogramCount(h)
	if count != 1 {
		t.Fatalf("expected 1 observation, got %d", count)
	}
}

func TestMemoryBytes(t *testing.T) {
	MemoryBytes.Set(1024 * 1024)
	if v := getGaugeValue(MemoryBytes); v != 1024*1024 {
		t.Fatalf("expected %v, got %v", 1024*1024, v)
	}
}

func TestIndexBuildDurationSeconds(t *testing.T) {
	start := time.Now()
	// Simulate a short build
	time.Sleep(time.Millisecond)
	IndexBuildDurationSeconds.Observe(time.Since(start).Seconds())

	var m dto.Metric
	IndexBuildDurationSeconds.Write(&m)
	if m.GetHistogram().GetSampleCount() != 1 {
		t.Fatalf("expected 1 observation, got %d", m.GetHistogram().GetSampleCount())
	}
}
