package client

import (
	"context"
	"math"
	"math/rand/v2"
	"time"
)

// RetryConfig controls retry behavior, matching the TS/Python SDKs.
type RetryConfig struct {
	MaxRetries    int           // Maximum retry attempts (default: 3)
	InitialDelay  time.Duration // First retry delay (default: 500ms)
	MaxDelay      time.Duration // Cap on delay (default: 30s)
	Multiplier    float64       // Backoff multiplier (default: 2.0)
	JitterPercent float64       // Jitter as fraction of delay (default: 0.1)
}

// DefaultRetryConfig returns defaults matching the TS/Python SDKs.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:    3,
		InitialDelay:  500 * time.Millisecond,
		MaxDelay:      30 * time.Second,
		Multiplier:    2.0,
		JitterPercent: 0.1,
	}
}

// WithRetry sets the retry configuration. Pass nil to disable retries.
func WithRetry(cfg *RetryConfig) Option {
	return func(c *Client) {
		c.retry = cfg
	}
}

// shouldRetry checks if the error is retryable and we haven't exhausted attempts.
func shouldRetry(err error, attempt int, cfg *RetryConfig) bool {
	if cfg == nil || attempt >= cfg.MaxRetries {
		return false
	}
	return IsRetryable(err)
}

// retryDelay computes the delay for the given attempt with exponential backoff + jitter.
func retryDelay(attempt int, cfg *RetryConfig) time.Duration {
	delay := float64(cfg.InitialDelay) * math.Pow(cfg.Multiplier, float64(attempt))
	if delay > float64(cfg.MaxDelay) {
		delay = float64(cfg.MaxDelay)
	}
	// Add jitter: ±JitterPercent
	jitter := delay * cfg.JitterPercent * (2*rand.Float64() - 1)
	d := time.Duration(delay + jitter)
	if d < 0 {
		d = time.Duration(delay)
	}
	return d
}

// sleepWithContext sleeps for d or returns early if ctx is cancelled.
func sleepWithContext(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
