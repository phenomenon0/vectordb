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

func newRateLimiter(rate, burst, maxBuckets int, intvl time.Duration) *rateLimiter {
	if maxBuckets <= 0 {
		maxBuckets = 100_000
	}
	rl := &rateLimiter{
		rate:       rate,
		burst:      burst,
		intvl:      intvl,
		buckets:    make(map[string]*bucket),
		maxBuckets: maxBuckets,
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

	rl.refillLocked(b, now)

	if b.tokens <= 0 {
		return false
	}
	b.tokens--
	return true
}

func (rl *rateLimiter) refillLocked(b *bucket, now time.Time) {
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
}

// failureBlocked checks whether a peer has exhausted its failed-auth budget
// without consuming a token. Unseen peers are admitted unless the bounded key
// map is full, in which case authentication fails closed without allocating.
func (rl *rateLimiter) failureBlocked(key string) bool {
	if rl == nil {
		return false
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, ok := rl.buckets[key]
	if !ok {
		return len(rl.buckets) >= rl.maxBuckets
	}
	rl.refillLocked(b, time.Now())
	return b.tokens <= 0
}

// recordFailure consumes one token only after credential verification fails.
// Successful authentication never calls this method and therefore never
// consumes the failed-auth budget.
func (rl *rateLimiter) recordFailure(key string) {
	if rl == nil {
		return
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	b, ok := rl.buckets[key]
	if !ok {
		if len(rl.buckets) >= rl.maxBuckets {
			return
		}
		b = &bucket{tokens: rl.burst, lastFill: now}
		rl.buckets[key] = b
	} else {
		rl.refillLocked(b, now)
	}
	if b.tokens > 0 {
		b.tokens--
	}
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
