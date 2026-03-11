package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/crc64"
	"io"
	"net/http"
	"sync"
	"time"
)

// Snapshot represents a full shard snapshot for sync
type Snapshot struct {
	ShardID   int             `json:"shard_id"`
	Sequence  uint64          `json:"sequence"`
	Timestamp time.Time       `json:"timestamp"`
	Vectors   []SnapshotEntry `json:"vectors"`
	Checksum  uint64          `json:"checksum"`
}

// SnapshotEntry represents a single vector in the snapshot
type SnapshotEntry struct {
	ID       string    `json:"id"`
	Vector   []float32 `json:"vector"`
	Metadata string    `json:"metadata,omitempty"`
}

// computeChecksum computes CRC64 checksum for snapshot validation
func (s *Snapshot) computeChecksum() uint64 {
	table := crc64.MakeTable(crc64.ECMA)
	h := crc64.New(table)

	// Hash shard ID and sequence
	fmt.Fprintf(h, "%d:%d:", s.ShardID, s.Sequence)

	// Hash each vector
	for _, entry := range s.Vectors {
		fmt.Fprintf(h, "%s:", entry.ID)
		for _, v := range entry.Vector {
			fmt.Fprintf(h, "%.6f,", v)
		}
	}

	return h.Sum64()
}

// validate checks if snapshot checksum matches
func (s *Snapshot) validate() bool {
	return s.Checksum == s.computeChecksum()
}

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

	// PollInterval is how often to check for new WAL entries when caught up
	PollInterval time.Duration

	// FastPollInterval is how often to poll when entries are available (adaptive mode).
	// Set to 0 to disable adaptive polling (use PollInterval always).
	FastPollInterval time.Duration

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
		PollInterval:         1 * time.Second,         // Poll interval when caught up
		FastPollInterval:     100 * time.Millisecond,  // Poll interval when entries available
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

// replicationLoop is the main replication loop with adaptive polling.
// When entries are available, polls at FastPollInterval for low latency.
// When caught up, polls at PollInterval to reduce overhead.
func (fr *FollowerReplicator) replicationLoop() {
	defer fr.wg.Done()

	reconnectAttempts := 0
	currentInterval := fr.config.PollInterval
	pollTicker := time.NewTicker(currentInterval)
	defer pollTicker.Stop()

	for {
		select {
		case <-fr.ctx.Done():
			return

		case <-pollTicker.C:
			hadEntries, err := fr.pollAdaptive()
			if err != nil {
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

				// Adaptive poll interval: fast when catching up, slow when caught up
				newInterval := fr.config.PollInterval
				if hadEntries && fr.config.FastPollInterval > 0 {
					newInterval = fr.config.FastPollInterval
				}
				if newInterval != currentInterval {
					currentInterval = newInterval
					pollTicker.Reset(currentInterval)
				}
			}
		}
	}
}

// pollAdaptive wraps poll and returns whether entries were received.
func (fr *FollowerReplicator) pollAdaptive() (hadEntries bool, err error) {
	before := fr.getAppliedCount()
	if err := fr.poll(); err != nil {
		return false, err
	}
	return fr.getAppliedCount() > before, nil
}

func (fr *FollowerReplicator) getAppliedCount() uint64 {
	fr.mu.RLock()
	defer fr.mu.RUnlock()
	return fr.stats.EntriesApplied
}

// poll fetches and applies new WAL entries
func (fr *FollowerReplicator) poll() error {
	start := time.Now()

	// Pull latest entries from primary
	entries, err := fr.walClient.PullLatest()
	if err != nil {
		// Check if error indicates we're too far behind (410 Gone)
		if fr.config.FullSyncThreshold > 0 {
			// Try full sync if WAL entries are trimmed
			fmt.Printf("[FollowerReplicator] WAL entries too old, attempting full snapshot sync\n")
			if syncErr := fr.RequestFullSync(fr.ctx); syncErr != nil {
				return fmt.Errorf("full sync failed: %w", syncErr)
			}
			return nil
		}
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

	// Check if lag exceeds threshold - trigger full sync
	lag := fr.GetLag()
	if fr.config.FullSyncThreshold > 0 && lag > fr.config.FullSyncThreshold {
		fmt.Printf("[FollowerReplicator] Lag %d exceeds threshold %d, triggering full sync\n",
			lag, fr.config.FullSyncThreshold)
		if syncErr := fr.RequestFullSync(fr.ctx); syncErr != nil {
			return fmt.Errorf("full sync failed: %w", syncErr)
		}
		return nil
	}

	// We're catching up
	fr.setStatus(StatusSyncing)

	batchSize := len(entries)
	if fr.config.BatchSize > 0 && batchSize > fr.config.BatchSize {
		batchSize = fr.config.BatchSize
	}

	var totalApplied uint64
	var totalApplyDuration time.Duration
	for startIdx := 0; startIdx < len(entries); startIdx += batchSize {
		endIdx := startIdx + batchSize
		if endIdx > len(entries) {
			endIdx = len(entries)
		}
		batch := entries[startIdx:endIdx]

		applyStart := time.Now()
		lastApplied, err := fr.shard.ApplyEntries(batch)
		totalApplyDuration += time.Since(applyStart)

		// FIX #3: Always advance the cursor to the last successfully applied
		// entry, even on partial failure. This prevents the wedge where retry
		// replays already-applied entries and gets stuck on duplicate-ID errors.
		if lastApplied > 0 {
			fr.walClient.Advance(lastApplied)
			fr.mu.Lock()
			fr.lastAppliedSeq = lastApplied
			fr.mu.Unlock()
		}

		if err != nil {
			// Count only the entries that were actually applied
			applied := uint64(0)
			for _, e := range batch {
				if e.Seq <= lastApplied {
					applied++
				}
			}
			totalApplied += applied
			return fmt.Errorf("apply failed at seq %d (advanced cursor to %d): %w", lastApplied+1, lastApplied, err)
		}
		totalApplied += uint64(len(batch))
	}

	fr.mu.Lock()
	fr.stats.EntriesApplied += totalApplied
	fr.stats.LastApplyDuration = totalApplyDuration
	fr.mu.Unlock()

	fr.setStatus(StatusStreaming)

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

	return replicationLag(fr.primarySeq, fr.lastAppliedSeq)
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
		"lag":                 replicationLag(fr.primarySeq, fr.lastAppliedSeq),
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
	fmt.Printf("[FollowerReplicator] Starting full snapshot sync from %s\n", fr.config.PrimaryAddr)

	// 1. Set status to syncing
	fr.setStatus(StatusSyncing)

	// 2. Request snapshot from primary
	snapshotURL := fmt.Sprintf("%s/internal/snapshot", fr.config.PrimaryAddr)
	req, err := http.NewRequestWithContext(ctx, "GET", snapshotURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create snapshot request: %w", err)
	}

	if fr.config.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+fr.config.AuthToken)
	}

	client := &http.Client{Timeout: 5 * time.Minute} // Snapshots can be large
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to request snapshot: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("snapshot request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// 3. Parse snapshot
	var snapshot Snapshot
	if err := json.NewDecoder(resp.Body).Decode(&snapshot); err != nil {
		return fmt.Errorf("failed to decode snapshot: %w", err)
	}

	// 4. Validate checksum
	if !snapshot.validate() {
		return fmt.Errorf("snapshot checksum mismatch: data corruption detected")
	}

	fmt.Printf("[FollowerReplicator] Received snapshot: seq=%d, vectors=%d\n",
		snapshot.Sequence, len(snapshot.Vectors))

	// 5. Load snapshot into local store
	if err := fr.shard.LoadSnapshot(&snapshot); err != nil {
		return fmt.Errorf("failed to load snapshot: %w", err)
	}

	// 6. Update replication state
	fr.mu.Lock()
	fr.lastAppliedSeq = snapshot.Sequence
	if snapshot.Sequence > fr.primarySeq {
		fr.primarySeq = snapshot.Sequence
	}
	fr.stats.EntriesApplied += uint64(len(snapshot.Vectors))
	fr.mu.Unlock()
	fr.walClient.Advance(snapshot.Sequence)

	fmt.Printf("[FollowerReplicator] Full sync complete, resuming from seq=%d\n", snapshot.Sequence)

	// 7. Status will transition to streaming on next successful poll
	return nil
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
	if replicationLag(fr.primarySeq, fr.lastAppliedSeq) > 10000 {
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

func replicationLag(primarySeq uint64, lastAppliedSeq uint64) uint64 {
	if primarySeq <= lastAppliedSeq {
		return 0
	}
	return primarySeq - lastAppliedSeq
}
