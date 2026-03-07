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

// ===========================================================================================
// PER-TENANT RATE LIMITING
// ===========================================================================================

type tenantRateLimiter struct {
	mu         sync.RWMutex
	limiters   map[string]*rateLimiter // tenantID -> rate limiter
	lastAccess map[string]time.Time    // tenantID -> last access time
	rps        int
	burst      int
	window     time.Duration
	maxTenants int           // Maximum number of tenants to prevent memory exhaustion
	cleanupAge time.Duration // Age after which inactive tenants are cleaned
}

func newTenantRateLimiter(rps, burst int, window time.Duration) *tenantRateLimiter {
	trl := &tenantRateLimiter{
		limiters:   make(map[string]*rateLimiter),
		lastAccess: make(map[string]time.Time),
		rps:        rps,
		burst:      burst,
		window:     window,
		maxTenants: 10000,            // Limit to 10k tenants
		cleanupAge: 30 * time.Minute, // Clean up tenants inactive for 30 minutes
	}
	// Start background cleanup
	go trl.cleanupLoop()
	return trl
}

func (trl *tenantRateLimiter) allow(tenantID string) bool {
	now := time.Now()

	trl.mu.RLock()
	limiter, ok := trl.limiters[tenantID]
	trl.mu.RUnlock()

	if !ok {
		trl.mu.Lock()
		// Double-check after acquiring write lock
		limiter, ok = trl.limiters[tenantID]
		if !ok {
			// Enforce tenant limit to prevent memory exhaustion
			if len(trl.limiters) >= trl.maxTenants {
				trl.mu.Unlock()
				return false // At capacity, deny new tenants
			}
			limiter = newRateLimiter(trl.rps, trl.burst, trl.window)
			trl.limiters[tenantID] = limiter
			trl.lastAccess[tenantID] = now
		}
		trl.mu.Unlock()
	} else {
		// Update last access time (write lock needed)
		trl.mu.Lock()
		trl.lastAccess[tenantID] = now
		trl.mu.Unlock()
	}

	return limiter.allow(tenantID)
}

func (trl *tenantRateLimiter) setLimit(tenantID string, rps, burst int) {
	trl.mu.Lock()
	defer trl.mu.Unlock()
	trl.limiters[tenantID] = newRateLimiter(rps, burst, trl.window)
	trl.lastAccess[tenantID] = time.Now()
}

// cleanupLoop periodically removes inactive tenant limiters
func (trl *tenantRateLimiter) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		trl.cleanup()
	}
}

// cleanup removes tenant limiters that haven't been accessed recently
func (trl *tenantRateLimiter) cleanup() {
	trl.mu.Lock()
	defer trl.mu.Unlock()

	cutoff := time.Now().Add(-trl.cleanupAge)
	for tenantID, lastAccess := range trl.lastAccess {
		if lastAccess.Before(cutoff) {
			delete(trl.limiters, tenantID)
			delete(trl.lastAccess, tenantID)
		}
	}
}
