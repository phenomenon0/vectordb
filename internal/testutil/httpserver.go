package testutil

import (
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"syscall"
	"testing"
)

// NewLoopbackServer starts an HTTP test server on IPv4 loopback and skips
// cleanly when the environment forbids opening local listeners.
func NewLoopbackServer(t testing.TB, handler http.Handler) *httptest.Server {
	t.Helper()

	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		if isListenerPermissionError(err) {
			t.Skipf("skipping listener-based test in restricted environment: %v", err)
		}
		t.Fatalf("listen on IPv4 loopback: %v", err)
	}

	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

func isListenerPermissionError(err error) bool {
	if errors.Is(err, syscall.EPERM) || errors.Is(err, syscall.EACCES) {
		return true
	}

	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "operation not permitted") || strings.Contains(msg, "permission denied")
}
