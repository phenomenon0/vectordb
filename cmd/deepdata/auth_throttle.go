package main

import (
	"context"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc/peer"
)

const unknownAuthPeerKey = "ip:unknown"

const authFailureLockStripes = 256

// authFailureLimiter serializes credential verification by a fixed set of
// peer-key stripes. This closes the check/record race without allocating state
// for successful peers; only a completed authentication failure creates or
// consumes a token bucket.
type authFailureLimiter struct {
	failures *rateLimiter
	stripes  [authFailureLockStripes]sync.Mutex
}

type authFailureAttempt struct {
	limiter *authFailureLimiter
	key     string
	stripe  *sync.Mutex
}

func newAuthFailureLimiter(rate, burst, maxBuckets int, interval time.Duration) *authFailureLimiter {
	return &authFailureLimiter{failures: newRateLimiter(rate, burst, maxBuckets, interval)}
}

func (limiter *authFailureLimiter) begin(key string) (*authFailureAttempt, bool) {
	if limiter == nil {
		return nil, true
	}
	stripe := &limiter.stripes[authFailureStripe(key)]
	stripe.Lock()
	if limiter.failures.failureBlocked(key) {
		stripe.Unlock()
		return nil, false
	}
	return &authFailureAttempt{limiter: limiter, key: key, stripe: stripe}, true
}

func (attempt *authFailureAttempt) finish(failed bool) {
	if attempt == nil {
		return
	}
	if failed {
		attempt.limiter.failures.recordFailure(attempt.key)
	}
	attempt.stripe.Unlock()
}

func authFailureStripe(key string) int {
	var hash uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		hash ^= uint32(key[i])
		hash *= 16777619
	}
	return int(hash % authFailureLockStripes)
}

// httpAuthPeerKey trusts forwarding headers only when the deployment has
// explicitly opted into TRUST_PROXY. Invalid forwarded values fall back to the
// socket peer so an untrusted string cannot create arbitrary limiter keys.
func httpAuthPeerKey(r *http.Request, trustProxy bool) string {
	if r == nil {
		return unknownAuthPeerKey
	}
	if trustProxy {
		if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
			if comma := strings.IndexByte(forwarded, ','); comma >= 0 {
				forwarded = forwarded[:comma]
			}
			if key, ok := canonicalIPPeerKey(forwarded); ok {
				return key
			}
			if key, ok := canonicalIPPeerKey(r.RemoteAddr); ok {
				return key
			}
			return unknownAuthPeerKey
		}
		if key, ok := canonicalIPPeerKey(r.Header.Get("X-Real-IP")); ok {
			return key
		}
	}
	if key, ok := canonicalIPPeerKey(r.RemoteAddr); ok {
		return key
	}
	return unknownAuthPeerKey
}

func grpcAuthPeerKey(ctx context.Context) string {
	requestPeer, ok := peer.FromContext(ctx)
	if !ok || requestPeer == nil || requestPeer.Addr == nil {
		return unknownAuthPeerKey
	}
	if tcpAddr, ok := requestPeer.Addr.(*net.TCPAddr); ok && tcpAddr.IP != nil {
		return "ip:" + tcpAddr.IP.String()
	}
	if key, ok := canonicalIPPeerKey(requestPeer.Addr.String()); ok {
		return key
	}
	return unknownAuthPeerKey
}

func canonicalIPPeerKey(rawAddress string) (string, bool) {
	address := strings.TrimSpace(rawAddress)
	if address == "" {
		return "", false
	}
	host := address
	if splitHost, _, err := net.SplitHostPort(address); err == nil {
		host = splitHost
	}
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if zone := strings.LastIndexByte(host, '%'); zone >= 0 {
		host = host[:zone]
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return "", false
	}
	return "ip:" + ip.String(), true
}
