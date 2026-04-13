package main

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	// Initialize global metrics so tests that call newHTTPHandler()
	// don't panic on nil globalMetrics. In production this is called
	// from main() before the server starts.
	initMetrics()

	os.Exit(m.Run())
}
