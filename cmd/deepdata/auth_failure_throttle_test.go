package main

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/security"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

const authThrottleTestPeerIP = "198.51.100.27"

func authThrottleHTTPRequest(handler http.Handler, token string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodGet, "/v3/tenants/acme", nil)
	request.RemoteAddr = net.JoinHostPort(authThrottleTestPeerIP, "41234")
	// An untrusted client must not be able to split its failure history by
	// changing forwarding headers.
	request.Header.Set("X-Forwarded-For", "203.0.113.99")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func authThrottleGRPCContext(token string) context.Context {
	ctx := context.Background()
	if token != "" {
		ctx = metadata.NewIncomingContext(ctx, metadata.Pairs("authorization", "Bearer "+token))
	}
	return peer.NewContext(ctx, &peer.Peer{
		Addr: &net.TCPAddr{IP: net.ParseIP(authThrottleTestPeerIP), Port: 50051},
	})
}

func TestFailedAuthenticationThrottleIsSharedAcrossHTTPAndGRPC(t *testing.T) {
	for _, authMode := range []string{"jwt", "static"} {
		t.Run(authMode, func(t *testing.T) {
			t.Setenv("JWT_SECRET", "")
			t.Setenv("API_TOKEN", "")
			t.Setenv("REQUIRE_AUTH", "0")
			t.Setenv("TRUST_PROXY", "0")
			t.Setenv("MAX_TENANTS", "10")
			t.Setenv("MAX_COLLECTIONS", "10")

			rt := testServerRuntime(t)
			rt.requireAuth = true
			validToken := ""
			switch authMode {
			case "jwt":
				rt.jwtMgr = security.NewJWTManager("failed-auth-throttle-jwt-secret", "failed-auth-test")
				var err error
				validToken, err = rt.jwtMgr.GenerateTenantToken("acme", []string{"admin"}, nil, time.Hour)
				if err != nil {
					t.Fatal(err)
				}
			case "static":
				rt.apiToken = "failed-auth-throttle-static-token"
			default:
				t.Fatalf("unknown auth mode %q", authMode)
			}
			if authMode == "static" {
				validToken = rt.apiToken
			}

			rt.rl = newRateLimiter(100, 100, 100, time.Hour)
			rt.canonicalTenantRL = newRateLimiter(100, 100, 100, time.Hour)
			rt.authFailureRL = newAuthFailureLimiter(1, 2, 100, time.Hour)
			handler, collections := newCanonicalHTTPHandler(
				rt,
				NewHashEmbedder(4),
				filepath.Join(t.TempDir(), "index.gob"),
			)
			t.Cleanup(func() { _ = collections.Close() })

			newInterceptor := func() func(context.Context, any, *grpc.UnaryServerInfo, grpc.UnaryHandler) (any, error) {
				return grpcAuthInterceptorWithRateLimiters(
					rt.jwtMgr,
					rt.apiToken,
					rt.requireAuth,
					testLogger(),
					nil,
					rt.authFailureRL,
				)
			}

			// One HTTP and one gRPC verification failure exhaust the same peer
			// bucket. Both transports then reject before credential verification.
			if response := authThrottleHTTPRequest(handler, "invalid-http-token"); response.Code != http.StatusUnauthorized {
				t.Fatalf("first HTTP failure returned %d: %s", response.Code, response.Body.String())
			}
			interceptor := newInterceptor()
			if _, err := interceptor(authThrottleGRPCContext("invalid-grpc-token"), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); status.Code(err) != codes.Unauthenticated {
				t.Fatalf("second gRPC failure = %v, want Unauthenticated", err)
			}
			if response := authThrottleHTTPRequest(handler, "still-invalid"); response.Code != http.StatusTooManyRequests {
				t.Fatalf("throttled HTTP failure returned %d: %s", response.Code, response.Body.String())
			}
			if _, err := interceptor(authThrottleGRPCContext("still-invalid"), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); status.Code(err) != codes.ResourceExhausted {
				t.Fatalf("throttled gRPC failure = %v, want ResourceExhausted", err)
			}

			// Reset only the test limiter. Successful authentication on either
			// transport does not consume its two allowed failure tokens.
			rt.authFailureRL = newAuthFailureLimiter(1, 2, 100, time.Hour)
			interceptor = newInterceptor()
			for i := 0; i < 3; i++ {
				if response := authThrottleHTTPRequest(handler, validToken); response.Code != http.StatusOK {
					t.Fatalf("valid HTTP authentication %d returned %d: %s", i, response.Code, response.Body.String())
				}
				if _, err := interceptor(authThrottleGRPCContext(validToken), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); err != nil {
					t.Fatalf("valid gRPC authentication %d failed: %v", i, err)
				}
			}
			if response := authThrottleHTTPRequest(handler, "invalid-after-success"); response.Code != http.StatusUnauthorized {
				t.Fatalf("first failure after successes returned %d: %s", response.Code, response.Body.String())
			}
			if response := authThrottleHTTPRequest(handler, validToken); response.Code != http.StatusOK {
				t.Fatalf("valid HTTP authentication consumed failure budget: %d: %s", response.Code, response.Body.String())
			}
			if _, err := interceptor(authThrottleGRPCContext(validToken), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); err != nil {
				t.Fatalf("valid gRPC authentication consumed failure budget: %v", err)
			}
			if _, err := interceptor(authThrottleGRPCContext("second-invalid"), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); status.Code(err) != codes.Unauthenticated {
				t.Fatalf("second failure after successes = %v, want Unauthenticated", err)
			}
			if response := authThrottleHTTPRequest(handler, "third-invalid"); response.Code != http.StatusTooManyRequests {
				t.Fatalf("post-budget HTTP request returned %d: %s", response.Code, response.Body.String())
			}

			// Missing credentials are failures in both implementations, not a
			// bypass around the failed-verification accounting.
			rt.authFailureRL = newAuthFailureLimiter(1, 1, 100, time.Hour)
			interceptor = newInterceptor()
			if response := authThrottleHTTPRequest(handler, ""); response.Code != http.StatusUnauthorized {
				t.Fatalf("missing HTTP credential returned %d: %s", response.Code, response.Body.String())
			}
			if _, err := interceptor(authThrottleGRPCContext(""), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); status.Code(err) != codes.ResourceExhausted {
				t.Fatalf("gRPC did not share missing HTTP failure: %v", err)
			}

			rt.authFailureRL = newAuthFailureLimiter(1, 1, 100, time.Hour)
			interceptor = newInterceptor()
			if _, err := interceptor(authThrottleGRPCContext(""), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler); status.Code(err) != codes.Unauthenticated {
				t.Fatalf("missing gRPC credential = %v, want Unauthenticated", err)
			}
			if response := authThrottleHTTPRequest(handler, ""); response.Code != http.StatusTooManyRequests {
				t.Fatalf("HTTP did not share missing gRPC failure: %d: %s", response.Code, response.Body.String())
			}

			// Credential verification for one peer is serialized across the two
			// transports, so concurrent brute force cannot overshoot the burst.
			rt.authFailureRL = newAuthFailureLimiter(1, 1, 100, time.Hour)
			interceptor = newInterceptor()
			const workers = 32
			start := make(chan struct{})
			results := make(chan string, workers)
			var wait sync.WaitGroup
			for i := 0; i < workers; i++ {
				wait.Add(1)
				go func(i int) {
					defer wait.Done()
					<-start
					if i%2 == 0 {
						response := authThrottleHTTPRequest(handler, "concurrent-invalid")
						switch response.Code {
						case http.StatusUnauthorized:
							results <- "failure"
						case http.StatusTooManyRequests:
							results <- "throttled"
						default:
							results <- "unexpected-http"
						}
						return
					}
					_, err := interceptor(authThrottleGRPCContext("concurrent-invalid"), nil, dummyServerInfo("/test.Auth/Check"), passThroughHandler)
					switch status.Code(err) {
					case codes.Unauthenticated:
						results <- "failure"
					case codes.ResourceExhausted:
						results <- "throttled"
					default:
						results <- "unexpected-grpc"
					}
				}(i)
			}
			close(start)
			wait.Wait()
			close(results)
			failures, throttled := 0, 0
			for result := range results {
				switch result {
				case "failure":
					failures++
				case "throttled":
					throttled++
				default:
					t.Fatalf("concurrent authentication returned %s", result)
				}
			}
			if failures != 1 || throttled != workers-1 {
				t.Fatalf("concurrent attempts = %d verification failures and %d throttles, want 1 and %d", failures, throttled, workers-1)
			}
		})
	}
}

func TestAuthFailurePeerKeysAreConsistentAndProxyExplicit(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "/", nil)
	request.RemoteAddr = net.JoinHostPort(authThrottleTestPeerIP, "41234")
	request.Header.Set("X-Forwarded-For", "203.0.113.8, 10.0.0.1")
	request.Header.Set("X-Real-IP", "203.0.113.9")

	grpcKey := grpcAuthPeerKey(authThrottleGRPCContext(""))
	if got := httpAuthPeerKey(request, false); got != grpcKey {
		t.Fatalf("untrusted HTTP peer key = %q, gRPC key = %q", got, grpcKey)
	}
	if got := httpAuthPeerKey(request, true); got != "ip:203.0.113.8" {
		t.Fatalf("trusted forwarded peer key = %q, want ip:203.0.113.8", got)
	}

	request.Header.Set("X-Forwarded-For", "not-an-ip")
	if got := httpAuthPeerKey(request, true); got != "ip:"+authThrottleTestPeerIP {
		t.Fatalf("invalid forwarded-for did not fall back to socket peer: %q", got)
	}
	if got, ok := canonicalIPPeerKey("[fe80::1%eth0]:443"); !ok || got != "ip:fe80::1" {
		t.Fatalf("zone-qualified IPv6 peer key = %q, %v", got, ok)
	}
}

func TestAuthFailureLimiterIsFailureOnlyAndBounded(t *testing.T) {
	limiter := newAuthFailureLimiter(1, 1, 1, time.Hour)
	for i := 0; i < 10; i++ {
		attempt, allowed := limiter.begin("ip:198.51.100.1")
		if !allowed {
			t.Fatal("non-consuming failure check unexpectedly blocked a fresh peer")
		}
		attempt.finish(false)
	}
	if len(limiter.failures.buckets) != 0 {
		t.Fatalf("successful prechecks allocated %d failure buckets", len(limiter.failures.buckets))
	}

	attempt, allowed := limiter.begin("ip:198.51.100.1")
	if !allowed {
		t.Fatal("fresh peer was blocked before its first failure")
	}
	attempt.finish(true)
	if _, allowed := limiter.begin("ip:198.51.100.1"); allowed {
		t.Fatal("recorded failure did not exhaust burst-one peer")
	}
	if _, allowed := limiter.begin("ip:198.51.100.2"); allowed {
		t.Fatal("bounded limiter did not fail closed for an unseen peer at capacity")
	}
	if len(limiter.failures.buckets) != 1 {
		t.Fatalf("failure limiter bucket count = %d, want 1", len(limiter.failures.buckets))
	}
}

func TestAuthFailureEnvironmentFailsFast(t *testing.T) {
	for _, key := range []string{"AUTH_FAILURE_RPS", "AUTH_FAILURE_BURST"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, "0")
			_, errs := loadServerConfig(nil, os.Getenv)
			found := false
			for _, err := range errs {
				if strings.HasPrefix(err, key+"=") {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("loadServerConfig did not reject %s=0: %v", key, errs)
			}
		})
	}
}
