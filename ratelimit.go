package main

import (
	"sync"
	"time"
)

// Simple token bucket rate limiter per key.
type rateLimiter struct {
	mu      sync.Mutex
	rate    int           // tokens per interval
	burst   int           // max tokens
	intvl   time.Duration // interval
	buckets map[string]*bucket
}

type bucket struct {
	tokens   int
	lastFill time.Time
}

func newRateLimiter(rate, burst int, intvl time.Duration) *rateLimiter {
	return &rateLimiter{
		rate:    rate,
		burst:   burst,
		intvl:   intvl,
		buckets: make(map[string]*bucket),
	}
}

func (rl *rateLimiter) allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	b, ok := rl.buckets[key]
	now := time.Now()
	if !ok {
		rl.buckets[key] = &bucket{tokens: rl.burst - 1, lastFill: now}
		return true
	}

	// refill
	elapsed := now.Sub(b.lastFill)
	if elapsed >= rl.intvl {
		add := int(elapsed / rl.intvl * time.Duration(rl.rate))
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
