package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ===========================================================================================
// FOLLOWER REPLICATION
// Continuous WAL streaming from primary to replica with automatic reconnection
// ===========================================================================================

// FollowerReplicatorConfig configures the follower replication process
type FollowerReplicatorConfig struct {
	// PrimaryAddr is the HTTP address of the primary node
	PrimaryAddr string

	// AuthToken for WAL stream authentication
	AuthToken string

	// PollInterval is how often to check for new WAL entries
	PollInterval time.Duration

	// ReconnectInterval is how long to wait before reconnecting on failure
	ReconnectInterval time.Duration

	// MaxReconnectAttempts before giving up (-1 for infinite)
	MaxReconnectAttempts int

	// BatchSize limits entries applied per poll (0 = unlimited)
	BatchSize int

	// FullSyncThreshold - if this far behind, request full snapshot
	FullSyncThreshold uint64
}

// DefaultFollowerReplicatorConfig returns sensible defaults
func DefaultFollowerReplicatorConfig() FollowerReplicatorConfig {
	return FollowerReplicatorConfig{
		PollInterval:         100 * time.Millisecond, // Low latency replication
		ReconnectInterval:    5 * time.Second,
		MaxReconnectAttempts: -1, // Infinite retries
		BatchSize:            1000,
		FullSyncThreshold:    100000, // Request snapshot if 100K behind
	}
}

// FollowerReplicator continuously replicates WAL from primary to replica
type FollowerReplicator struct {
	mu sync.RWMutex

	config    FollowerReplicatorConfig
	shard     *ShardServer
	walClient *WALStreamClient

	// State
	lastAppliedSeq uint64
	primarySeq     uint64
	status         ReplicationStatus
	stats          FollowerStats

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ReplicationStatus represents the current replication state
type ReplicationStatus int

const (
	StatusDisconnected ReplicationStatus = iota
	StatusConnecting
	StatusSyncing   // Initial sync / catching up
	StatusStreaming // Real-time streaming
	StatusError
)

func (s ReplicationStatus) String() string {
	switch s {
	case StatusDisconnected:
		return "disconnected"
	case StatusConnecting:
		return "connecting"
	case StatusSyncing:
		return "syncing"
	case StatusStreaming:
		return "streaming"
	case StatusError:
		return "error"
	default:
		return "unknown"
	}
}

// FollowerStats tracks replication statistics
type FollowerStats struct {
	EntriesApplied    uint64
	BytesReceived     uint64
	LastApplyDuration time.Duration
	LastPollTime      time.Time
	ReconnectCount    int
	ErrorCount        int
	LastError         string
	LastErrorTime     time.Time
}

// NewFollowerReplicator creates a new follower replicator
func NewFollowerReplicator(shard *ShardServer, config FollowerReplicatorConfig) *FollowerReplicator {
	ctx, cancel := context.WithCancel(context.Background())

	return &FollowerReplicator{
		config:    config,
		shard:     shard,
		walClient: NewWALStreamClient(config.PrimaryAddr, config.AuthToken),
		status:    StatusDisconnected,
		ctx:       ctx,
		cancel:    cancel,
	}
}

// Start begins the replication process
func (fr *FollowerReplicator) Start() {
	fr.wg.Add(1)
	go fr.replicationLoop()

	fmt.Printf("[FollowerReplicator] Started replication from %s\n", fr.config.PrimaryAddr)
}

// Stop stops the replication process
func (fr *FollowerReplicator) Stop() {
	fmt.Printf("[FollowerReplicator] Stopping replication...\n")
	fr.cancel()
	fr.wg.Wait()
	fmt.Printf("[FollowerReplicator] Stopped\n")
}

// replicationLoop is the main replication loop
func (fr *FollowerReplicator) replicationLoop() {
	defer fr.wg.Done()

	reconnectAttempts := 0
	pollTicker := time.NewTicker(fr.config.PollInterval)
	defer pollTicker.Stop()

	for {
		select {
		case <-fr.ctx.Done():
			return

		case <-pollTicker.C:
			if err := fr.poll(); err != nil {
				fr.handleError(err)
				reconnectAttempts++

				// Check max reconnect attempts
				if fr.config.MaxReconnectAttempts >= 0 &&
					reconnectAttempts > fr.config.MaxReconnectAttempts {
					fmt.Printf("[FollowerReplicator] Max reconnect attempts exceeded, stopping\n")
					return
				}

				// Wait before retry
				fr.setStatus(StatusConnecting)
				select {
				case <-time.After(fr.config.ReconnectInterval):
				case <-fr.ctx.Done():
					return
				}
			} else {
				// Successful poll - reset reconnect counter
				reconnectAttempts = 0
			}
		}
	}
}

// poll fetches and applies new WAL entries
func (fr *FollowerReplicator) poll() error {
	start := time.Now()

	// Pull latest entries from primary
	entries, err := fr.walClient.PullLatest()
	if err != nil {
		return fmt.Errorf("pull failed: %w", err)
	}

	fr.mu.Lock()
	fr.stats.LastPollTime = start
	fr.mu.Unlock()

	if len(entries) == 0 {
		// No new entries - we're up to date
		fr.setStatus(StatusStreaming)
		return nil
	}

	// We're catching up
	fr.setStatus(StatusSyncing)

	// Apply entries in batches
	batch := entries
	if fr.config.BatchSize > 0 && len(entries) > fr.config.BatchSize {
		batch = entries[:fr.config.BatchSize]
	}

	applyStart := time.Now()
	if err := fr.shard.ApplyEntries(batch); err != nil {
		return fmt.Errorf("apply failed: %w", err)
	}
	applyDuration := time.Since(applyStart)

	// Update stats
	fr.mu.Lock()
	fr.stats.EntriesApplied += uint64(len(batch))
	fr.stats.LastApplyDuration = applyDuration
	if len(batch) > 0 {
		fr.lastAppliedSeq = batch[len(batch)-1].Seq
	}
	fr.mu.Unlock()

	// Check if we're now streaming (caught up)
	if len(entries) <= fr.config.BatchSize {
		fr.setStatus(StatusStreaming)
	}

	return nil
}

// handleError records and logs an error
func (fr *FollowerReplicator) handleError(err error) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	fr.stats.ErrorCount++
	fr.stats.LastError = err.Error()
	fr.stats.LastErrorTime = time.Now()
	fr.status = StatusError

	fmt.Printf("[FollowerReplicator] Error: %v (total errors: %d)\n",
		err, fr.stats.ErrorCount)
}

// setStatus updates the replication status
func (fr *FollowerReplicator) setStatus(status ReplicationStatus) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if fr.status != status {
		fmt.Printf("[FollowerReplicator] Status: %s -> %s\n",
			fr.status.String(), status.String())
		fr.status = status
	}
}

// GetStatus returns current replication status
func (fr *FollowerReplicator) GetStatus() ReplicationStatus {
	fr.mu.RLock()
	defer fr.mu.RUnlock()
	return fr.status
}

// GetLag returns the replication lag (operations behind primary)
func (fr *FollowerReplicator) GetLag() uint64 {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	if fr.primarySeq <= fr.lastAppliedSeq {
		return 0
	}
	return fr.primarySeq - fr.lastAppliedSeq
}

// GetStats returns replication statistics
func (fr *FollowerReplicator) GetStats() map[string]any {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	return map[string]any{
		"status":              fr.status.String(),
		"primary_addr":        fr.config.PrimaryAddr,
		"last_applied_seq":    fr.lastAppliedSeq,
		"primary_seq":         fr.primarySeq,
		"lag":                 fr.primarySeq - fr.lastAppliedSeq,
		"entries_applied":     fr.stats.EntriesApplied,
		"bytes_received":      fr.stats.BytesReceived,
		"last_apply_duration": fr.stats.LastApplyDuration.String(),
		"last_poll_time":      fr.stats.LastPollTime.Unix(),
		"reconnect_count":     fr.stats.ReconnectCount,
		"error_count":         fr.stats.ErrorCount,
		"last_error":          fr.stats.LastError,
	}
}

// SetPrimaryAddr updates the primary address (for failover)
func (fr *FollowerReplicator) SetPrimaryAddr(addr string, token string) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	fr.config.PrimaryAddr = addr
	fr.config.AuthToken = token
	fr.walClient = NewWALStreamClient(addr, token)
	fr.stats.ReconnectCount++

	fmt.Printf("[FollowerReplicator] Primary changed to %s\n", addr)
}

// RequestFullSync requests a full snapshot sync (when too far behind)
func (fr *FollowerReplicator) RequestFullSync(ctx context.Context) error {
	// TODO: Implement full snapshot sync
	// 1. Pause replication
	// 2. Request snapshot from primary
	// 3. Load snapshot into local store
	// 4. Resume replication from snapshot sequence
	return fmt.Errorf("full sync not implemented")
}

// ===========================================================================================
// INTEGRATION HELPERS
// ===========================================================================================

// StartFollowerReplication starts replication for a shard server
func StartFollowerReplication(shard *ShardServer, primaryAddr, authToken string) *FollowerReplicator {
	config := DefaultFollowerReplicatorConfig()
	config.PrimaryAddr = primaryAddr
	config.AuthToken = authToken

	replicator := NewFollowerReplicator(shard, config)
	replicator.Start()

	return replicator
}

// FollowerHealthCheck checks if replication is healthy
func (fr *FollowerReplicator) FollowerHealthCheck() bool {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	// Healthy if streaming and no recent errors
	if fr.status != StatusStreaming {
		return false
	}

	// Unhealthy if lag is too high
	if fr.primarySeq-fr.lastAppliedSeq > 10000 {
		return false
	}

	// Unhealthy if last poll was too long ago
	if time.Since(fr.stats.LastPollTime) > 30*time.Second {
		return false
	}

	return true
}

// UpdatePrimarySequence updates the known primary sequence (from heartbeat)
func (fr *FollowerReplicator) UpdatePrimarySequence(seq uint64) {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if seq > fr.primarySeq {
		fr.primarySeq = seq
	}
}
