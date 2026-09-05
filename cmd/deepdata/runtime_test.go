package main

import (
	"os"
	"testing"
)

// testServerRuntime builds a serverRuntime the same way the server does:
// load configuration from flags/environment, then construct the runtime
// from it. Callers set the environment with t.Setenv before calling this.
func testServerRuntime(t *testing.T) *serverRuntime {
	t.Helper()
	cfg, errs := loadServerConfig(nil, os.Getenv)
	if len(errs) > 0 {
		t.Fatalf("loadServerConfig: %v", errs)
	}
	return newServerRuntime(cfg)
}
