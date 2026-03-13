package cluster

import (
	"context"
	"sync"
	"testing"
	"time"
)

// ===========================================================================================
// FAILOVER MANAGER TESTS
// Tests for automatic failover, replica selection, and health monitoring
// ===========================================================================================

// mockDistributedVectorDB creates a minimal mock for testing failover
func mockDistributedVectorDB(shards map[int][]*ShardNode) *DistributedVectorDB {
	return &DistributedVectorDB{
		shards: shards,
	}
}

// TestNewFailoverManager tests failover manager creation
func TestNewFailoverManager(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))

	tests := []struct {
		name          string
		config        FailoverConfig
		wantThreshold time.Duration
		wantInterval  time.Duration
	}{
		{
			name:          "default config values",
			config:        FailoverConfig{},
			wantThreshold: 30 * time.Second,
			wantInterval:  5 * time.Second,
		},
		{
			name: "custom config values",
			config: FailoverConfig{
				UnhealthyThreshold: 60 * time.Second,
				CheckInterval:      10 * time.Second,
				EnableAutoFailover: true,
			},
			wantThreshold: 60 * time.Second,
			wantInterval:  10 * time.Second,
		},
		{
			name: "partial custom config",
			config: FailoverConfig{
				UnhealthyThreshold: 15 * time.Second,
			},
			wantThreshold: 15 * time.Second,
			wantInterval:  5 * time.Second, // default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fm := NewFailoverManager(coordinator, tt.config)
			if fm == nil {
				t.Fatal("expected non-nil FailoverManager")
			}
			if fm.config.UnhealthyThreshold != tt.wantThreshold {
				t.Errorf("UnhealthyThreshold = %v, want %v", fm.config.UnhealthyThreshold, tt.wantThreshold)
			}
			if fm.config.CheckInterval != tt.wantInterval {
				t.Errorf("CheckInterval = %v, want %v", fm.config.CheckInterval, tt.wantInterval)
			}
			if fm.shardStates == nil {
				t.Error("expected shardStates map to be initialized")
			}
			if fm.stopCh == nil {
				t.Error("expected stopCh to be initialized")
			}
		})
	}
}

// TestFailoverManagerStartStop tests start/stop lifecycle
func TestFailoverManagerStartStop(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))

	t.Run("start with auto-failover disabled", func(t *testing.T) {
		fm := NewFailoverManager(coordinator, FailoverConfig{
			EnableAutoFailover: false,
		})

		ctx := context.Background()
		err := fm.Start(ctx)
		if err != nil {
			t.Fatalf("Start() error = %v", err)
		}

		// Should return immediately without starting monitor loop
		fm.Stop()
	})

	t.Run("start and stop with auto-failover enabled", func(t *testing.T) {
		fm := NewFailoverManager(coordinator, FailoverConfig{
			EnableAutoFailover: true,
			CheckInterval:      100 * time.Millisecond,
		})

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		err := fm.Start(ctx)
		if err != nil {
			t.Fatalf("Start() error = %v", err)
		}

		// Let it run briefly
		time.Sleep(150 * time.Millisecond)

		// Stop should complete without hanging
		done := make(chan struct{})
		go func() {
			fm.Stop()
			close(done)
		}()

		select {
		case <-done:
			// Success
		case <-time.After(2 * time.Second):
			t.Fatal("Stop() timed out")
		}
	})

	t.Run("context cancellation stops monitor", func(t *testing.T) {
		fm := NewFailoverManager(coordinator, FailoverConfig{
			EnableAutoFailover: true,
			CheckInterval:      50 * time.Millisecond,
		})

		ctx, cancel := context.WithCancel(context.Background())
		err := fm.Start(ctx)
		if err != nil {
			t.Fatalf("Start() error = %v", err)
		}

		// Cancel context
		cancel()
		time.Sleep(100 * time.Millisecond)

		// Stop should complete quickly
		fm.Stop()
	})
}

// TestCalculateReplicaScore tests the replica scoring algorithm
func TestCalculateReplicaScore(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))
	fm := NewFailoverManager(coordinator, FailoverConfig{})
	now := time.Now()

	tests := []struct {
		name     string
		replica  *ShardNode
		minScore float64
		maxScore float64
	}{
		{
			name: "perfect replica - low lag, recent, replica role",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 0,
				LastSeen:       now,
			},
			minScore: 105, // 100 base + 10 for replica role - minimal penalties
			maxScore: 115,
		},
		{
			name: "high lag replica",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 1000, // Very high lag
				LastSeen:       now,
			},
			minScore: 55, // Capped at -50 for lag
			maxScore: 70,
		},
		{
			name: "stale replica - 60+ seconds",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 0,
				LastSeen:       now.Add(-70 * time.Second),
			},
			minScore: 75, // -30 for staleness
			maxScore: 85,
		},
		{
			name: "stale replica - 30-60 seconds",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 0,
				LastSeen:       now.Add(-45 * time.Second),
			},
			minScore: 90, // -15 for staleness
			maxScore: 100,
		},
		{
			name: "stale replica - 10-30 seconds",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 0,
				LastSeen:       now.Add(-20 * time.Second),
			},
			minScore: 100, // -5 for staleness
			maxScore: 110,
		},
		{
			name: "combined penalties - lag and staleness",
			replica: &ShardNode{
				Role:           RoleReplica,
				ReplicationLag: 500,
				LastSeen:       now.Add(-45 * time.Second),
			},
			minScore: 60, // -25 for lag, -15 for staleness + 10 for role
			maxScore: 80,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := fm.calculateReplicaScore(tt.replica, now)
			if score < tt.minScore || score > tt.maxScore {
				t.Errorf("score = %v, want between %v and %v", score, tt.minScore, tt.maxScore)
			}
		})
	}
}

// TestSelectBestReplica tests replica selection logic
func TestSelectBestReplica(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))
	fm := NewFailoverManager(coordinator, FailoverConfig{})
	now := time.Now()

	tests := []struct {
		name       string
		replicas   []*ShardNode
		wantNodeID string
		wantNil    bool
	}{
		{
			name:     "empty replicas",
			replicas: []*ShardNode{},
			wantNil:  true,
		},
		{
			name: "all unhealthy replicas",
			replicas: []*ShardNode{
				{NodeID: "replica-1", Healthy: false, Role: RoleReplica, LastSeen: now},
				{NodeID: "replica-2", Healthy: false, Role: RoleReplica, LastSeen: now},
			},
			wantNil: true,
		},
		{
			name: "single healthy replica",
			replicas: []*ShardNode{
				{NodeID: "replica-1", Healthy: true, Role: RoleReplica, ReplicationLag: 10, LastSeen: now},
			},
			wantNodeID: "replica-1",
		},
		{
			name: "select lowest lag replica",
			replicas: []*ShardNode{
				{NodeID: "replica-1", Healthy: true, Role: RoleReplica, ReplicationLag: 100, LastSeen: now},
				{NodeID: "replica-2", Healthy: true, Role: RoleReplica, ReplicationLag: 5, LastSeen: now},
				{NodeID: "replica-3", Healthy: true, Role: RoleReplica, ReplicationLag: 50, LastSeen: now},
			},
			wantNodeID: "replica-2",
		},
		{
			name: "prefer recent over stale with same lag",
			replicas: []*ShardNode{
				{NodeID: "replica-1", Healthy: true, Role: RoleReplica, ReplicationLag: 10, LastSeen: now.Add(-60 * time.Second)},
				{NodeID: "replica-2", Healthy: true, Role: RoleReplica, ReplicationLag: 10, LastSeen: now},
			},
			wantNodeID: "replica-2",
		},
		{
			name: "mixed healthy and unhealthy",
			replicas: []*ShardNode{
				{NodeID: "replica-1", Healthy: false, Role: RoleReplica, ReplicationLag: 0, LastSeen: now},
				{NodeID: "replica-2", Healthy: true, Role: RoleReplica, ReplicationLag: 50, LastSeen: now},
				{NodeID: "replica-3", Healthy: false, Role: RoleReplica, ReplicationLag: 0, LastSeen: now},
			},
			wantNodeID: "replica-2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fm.selectBestReplica(tt.replicas)
			if tt.wantNil {
				if result != nil {
					t.Errorf("expected nil, got %v", result.NodeID)
				}
				return
			}
			if result == nil {
				t.Fatal("expected non-nil result")
			}
			if result.NodeID != tt.wantNodeID {
				t.Errorf("selected %v, want %v", result.NodeID, tt.wantNodeID)
			}
		})
	}
}

// TestGetShardState tests shard state retrieval
func TestGetShardState(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))
	fm := NewFailoverManager(coordinator, FailoverConfig{})

	// Initially no state
	state := fm.GetShardState(0)
	if state != nil {
		t.Error("expected nil state for non-existent shard")
	}

	// Add state manually
	fm.mu.Lock()
	fm.shardStates[0] = &shardFailoverState{
		shardID:            0,
		failoverInProgress: false,
		lastCheckTime:      time.Now(),
	}
	fm.mu.Unlock()

	state = fm.GetShardState(0)
	if state == nil {
		t.Fatal("expected non-nil state")
	}
	if state.shardID != 0 {
		t.Errorf("shardID = %d, want 0", state.shardID)
	}
}

// TestGetFailoverStats tests statistics retrieval
func TestGetFailoverStats(t *testing.T) {
	coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))
	config := FailoverConfig{
		EnableAutoFailover: true,
		UnhealthyThreshold: 45 * time.Second,
		CheckInterval:      10 * time.Second,
	}
	fm := NewFailoverManager(coordinator, config)

	// Add some shard states
	now := time.Now()
	unhealthySince := now.Add(-30 * time.Second)
	fm.mu.Lock()
	fm.shardStates[0] = &shardFailoverState{
		shardID:               0,
		primaryUnhealthySince: &unhealthySince,
		failoverInProgress:    false,
		lastCheckTime:         now,
	}
	fm.shardStates[1] = &shardFailoverState{
		shardID:            1,
		failoverInProgress: true,
		lastCheckTime:      now,
	}
	fm.mu.Unlock()

	stats := fm.GetFailoverStats()

	// Check config values
	if enabled, ok := stats["enabled"].(bool); !ok || !enabled {
		t.Error("expected enabled = true")
	}
	if threshold, ok := stats["unhealthy_threshold"].(string); !ok || threshold != "45s" {
		t.Errorf("unhealthy_threshold = %v, want 45s", threshold)
	}
	if interval, ok := stats["check_interval"].(string); !ok || interval != "10s" {
		t.Errorf("check_interval = %v, want 10s", interval)
	}

	// Check shard stats
	shards, ok := stats["shards"].([]map[string]any)
	if !ok {
		t.Fatal("expected shards to be []map[string]any")
	}
	if len(shards) != 2 {
		t.Errorf("expected 2 shards, got %d", len(shards))
	}
}

// TestManualFailover tests manual failover triggering
func TestManualFailover(t *testing.T) {
	now := time.Now()

	t.Run("shard not found", func(t *testing.T) {
		coordinator := mockDistributedVectorDB(make(map[int][]*ShardNode))
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		err := fm.ManualFailover(99)
		if err == nil {
			t.Error("expected error for non-existent shard")
		}
	})

	t.Run("no primary found", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "replica-1", Role: RoleReplica, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		err := fm.ManualFailover(0)
		if err == nil {
			t.Error("expected error for missing primary")
		}
	})

	t.Run("no replicas available", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", Role: RolePrimary, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		err := fm.ManualFailover(0)
		if err == nil {
			t.Error("expected error for missing replicas")
		}
	})

	t.Run("successful manual failover initiation", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
				{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now, ReplicationLag: 5},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		err := fm.ManualFailover(0)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		// Wait a bit for async failover to start
		time.Sleep(50 * time.Millisecond)

		// Check state was created
		state := fm.GetShardState(0)
		if state == nil {
			t.Error("expected state to be created")
		}
	})

	t.Run("rejects second manual trigger while in progress", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
				{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})
		fm.shardStates[0] = &shardFailoverState{
			shardID:            0,
			failoverInProgress: true,
		}

		err := fm.ManualFailover(0)
		if err == nil {
			t.Fatal("expected error when manual failover is already in progress")
		}
	})
}

// TestPromoteReplica tests replica promotion
func TestPromoteReplica(t *testing.T) {
	now := time.Now()

	t.Run("successful promotion", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
				{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		replica := shards[0][1]
		err := fm.promoteReplica(replica)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify role changed
		if replica.Role != RolePrimary {
			t.Errorf("replica role = %v, want %v", replica.Role, RolePrimary)
		}
	})

	t.Run("replica not found in coordinator", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{})

		// Replica not in coordinator's shard list
		replica := &ShardNode{NodeID: "unknown-replica", ShardID: 0, Role: RoleReplica}
		err := fm.promoteReplica(replica)
		if err == nil {
			t.Error("expected error for unknown replica")
		}
	})
}

// TestDemotePrimary tests primary demotion
func TestDemotePrimary(t *testing.T) {
	now := time.Now()

	shards := map[int][]*ShardNode{
		0: {
			{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: false, LastSeen: now},
			{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now},
		},
	}
	coordinator := mockDistributedVectorDB(shards)
	fm := NewFailoverManager(coordinator, FailoverConfig{})

	oldPrimary := shards[0][0]
	fm.demotePrimary(oldPrimary)

	if oldPrimary.Role != RoleReplica {
		t.Errorf("old primary role = %v, want %v", oldPrimary.Role, RoleReplica)
	}
}

// TestCheckShardHealthTransitions tests health state transitions
func TestCheckShardHealthTransitions(t *testing.T) {
	now := time.Now()

	t.Run("primary healthy - no state change", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{
			UnhealthyThreshold: 30 * time.Second,
		})

		fm.checkShard(0, shards[0])

		state := fm.GetShardState(0)
		if state == nil {
			t.Fatal("expected state to be created")
		}
		if state.primaryUnhealthySince != nil {
			t.Error("expected primaryUnhealthySince to be nil for healthy primary")
		}
	})

	t.Run("primary becomes unhealthy - timer starts", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: false, LastSeen: now},
				{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{
			UnhealthyThreshold: 30 * time.Second,
		})

		fm.checkShard(0, shards[0])

		state := fm.GetShardState(0)
		if state == nil {
			t.Fatal("expected state to be created")
		}
		if state.primaryUnhealthySince == nil {
			t.Error("expected primaryUnhealthySince to be set")
		}
	})

	t.Run("primary recovers - timer resets", func(t *testing.T) {
		shards := map[int][]*ShardNode{
			0: {
				{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
			},
		}
		coordinator := mockDistributedVectorDB(shards)
		fm := NewFailoverManager(coordinator, FailoverConfig{
			UnhealthyThreshold: 30 * time.Second,
		})

		// Set up as if primary was unhealthy
		unhealthySince := now.Add(-10 * time.Second)
		fm.mu.Lock()
		fm.shardStates[0] = &shardFailoverState{
			shardID:               0,
			primaryUnhealthySince: &unhealthySince,
		}
		fm.mu.Unlock()

		// Primary is now healthy
		fm.checkShard(0, shards[0])

		state := fm.GetShardState(0)
		if state.primaryUnhealthySince != nil {
			t.Error("expected primaryUnhealthySince to be reset")
		}
	})
}

// TestConcurrentFailoverOperations tests concurrent access safety
func TestConcurrentFailoverOperations(t *testing.T) {
	now := time.Now()
	shards := map[int][]*ShardNode{
		0: {
			{NodeID: "primary-1", ShardID: 0, Role: RolePrimary, Healthy: true, LastSeen: now},
			{NodeID: "replica-1", ShardID: 0, Role: RoleReplica, Healthy: true, LastSeen: now},
		},
		1: {
			{NodeID: "primary-2", ShardID: 1, Role: RolePrimary, Healthy: true, LastSeen: now},
			{NodeID: "replica-2", ShardID: 1, Role: RoleReplica, Healthy: true, LastSeen: now},
		},
	}
	coordinator := mockDistributedVectorDB(shards)
	fm := NewFailoverManager(coordinator, FailoverConfig{})

	var wg sync.WaitGroup
	operations := 100

	// Concurrent stats reads
	wg.Add(operations)
	for i := 0; i < operations; i++ {
		go func() {
			defer wg.Done()
			_ = fm.GetFailoverStats()
		}()
	}

	// Concurrent state reads
	wg.Add(operations)
	for i := 0; i < operations; i++ {
		go func(shardID int) {
			defer wg.Done()
			_ = fm.GetShardState(shardID % 2)
		}(i)
	}

	// Concurrent shard checks
	wg.Add(operations)
	for i := 0; i < operations; i++ {
		go func(shardID int) {
			defer wg.Done()
			fm.checkShard(shardID%2, shards[shardID%2])
		}(i)
	}

	wg.Wait()
}
