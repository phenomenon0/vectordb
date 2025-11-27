package main

import (
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	metricsOnce    sync.Once
	reqCounter     *prometheus.CounterVec
	reqDuration    *prometheus.HistogramVec
	obsGaugeActive prometheus.Gauge
)

func initMetrics() {
	metricsOnce.Do(func() {
		reqCounter = prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_requests_total",
				Help: "Total HTTP requests",
			},
			[]string{"path", "status"},
		)
		reqDuration = prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_request_duration_seconds",
				Help:    "HTTP request duration",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"path"},
		)
		obsGaugeActive = prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "vectordb_active_docs",
			Help: "Active documents (not tombstoned)",
		})
		prometheus.MustRegister(reqCounter, reqDuration, obsGaugeActive)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func withMetrics(path string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next(rec, r)
		if reqCounter != nil {
			reqCounter.WithLabelValues(path, strconv.Itoa(rec.status)).Inc()
		}
		if reqDuration != nil {
			reqDuration.WithLabelValues(path).Observe(time.Since(start).Seconds())
		}
	}
}
