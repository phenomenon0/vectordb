package main

import (
	"sync"
	"time"
)

// Simple token bucket rate limiter per key.
type rateLimiter struct {
	mu         sync.Mutex
	rate       int           // tokens per interval
	burst      int           // max tokens
	intvl      time.Duration // interval
	buckets    map[string]*bucket
	maxBuckets int           // max number of buckets to prevent memory exhaustion
	cleanupAge time.Duration // age after which inactive buckets are cleaned
}

type bucket struct {
	tokens   int
	lastFill time.Time
}

func newRateLimiter(rate, burst int, intvl time.Duration) *rateLimiter {
	rl := &rateLimiter{
		rate:       rate,
		burst:      burst,
		intvl:      intvl,
		buckets:    make(map[string]*bucket),
		maxBuckets: 100000,           // Limit to 100k unique keys
		cleanupAge: 10 * time.Minute, // Clean up buckets inactive for 10 minutes
	}
	// Start background cleanup
	go rl.cleanupLoop()
	return rl
}

func (rl *rateLimiter) allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, ok := rl.buckets[key]
	now := time.Now()
	if !ok {
		// Enforce bucket limit to prevent memory exhaustion
		if len(rl.buckets) >= rl.maxBuckets {
			// At capacity - deny new keys (they can retry after cleanup)
			return false
		}
		rl.buckets[key] = &bucket{tokens: rl.burst - 1, lastFill: now}
		return true
	}

	// refill
	elapsed := now.Sub(b.lastFill)
	if elapsed >= rl.intvl {
		// Prevent overflow with large elapsed times
		intervals := int(elapsed / rl.intvl)
		if intervals > rl.burst {
			intervals = rl.burst
		}
		add := intervals * rl.rate
		b.tokens += add
		if b.tokens > rl.burst {
			b.tokens = rl.burst
		}
		b.lastFill = now
	}

	if b.tokens <= 0 {
		return false
	}
	b.tokens--
	return true
}

// cleanupLoop periodically removes stale buckets
func (rl *rateLimiter) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		rl.cleanup()
	}
}

// cleanup removes buckets that haven't been used recently
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	cutoff := time.Now().Add(-rl.cleanupAge)
	for key, b := range rl.buckets {
		if b.lastFill.Before(cutoff) {
			delete(rl.buckets, key)
		}
	}
}
