package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ===========================================================================================
// AUTOMATIC FAILOVER MANAGER
// Monitors primary health and automatically promotes replicas when primary fails
// ===========================================================================================

// FailoverConfig configures the failover manager
type FailoverConfig struct {
	UnhealthyThreshold time.Duration // How long primary must be unhealthy before failover (default: 30s)
	CheckInterval      time.Duration // How often to check health (default: 5s)
	EnableAutoFailover bool          // Enable automatic failover (default: false for safety)
}

// FailoverManager handles automatic replica promotion when primary fails
type FailoverManager struct {
	mu sync.RWMutex

	coordinator *DistributedVectorDB
	config      FailoverConfig

	// Per-shard failover state
	shardStates map[int]*shardFailoverState

	// Control
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// shardFailoverState tracks failover state for a single shard
type shardFailoverState struct {
	shardID              int
	primaryUnhealthySince *time.Time // When primary became unhealthy (nil if healthy)
	failoverInProgress    bool
	lastCheckTime         time.Time
}

// NewFailoverManager creates a new failover manager
func NewFailoverManager(coordinator *DistributedVectorDB, config FailoverConfig) *FailoverManager {
	if config.UnhealthyThreshold == 0 {
		config.UnhealthyThreshold = 30 * time.Second
	}
	if config.CheckInterval == 0 {
		config.CheckInterval = 5 * time.Second
	}

	fm := &FailoverManager{
		coordinator: coordinator,
		config:      config,
		shardStates: make(map[int]*shardFailoverState),
		stopCh:      make(chan struct{}),
	}

	return fm
}

// Start starts the failover manager
func (fm *FailoverManager) Start(ctx context.Context) error {
	if !fm.config.EnableAutoFailover {
		fmt.Println("⚠️  Automatic failover is DISABLED (set EnableAutoFailover=true to enable)")
		return nil
	}

	fmt.Printf("✅ Automatic failover ENABLED (threshold: %v, check interval: %v)\n",
		fm.config.UnhealthyThreshold, fm.config.CheckInterval)

	fm.wg.Add(1)
	go fm.monitorLoop(ctx)

	return nil
}

// Stop stops the failover manager
func (fm *FailoverManager) Stop() {
	close(fm.stopCh)
	fm.wg.Wait()
}

// monitorLoop continuously monitors shard health and triggers failover
func (fm *FailoverManager) monitorLoop(ctx context.Context) {
	defer fm.wg.Done()

	ticker := time.NewTicker(fm.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-fm.stopCh:
			return
		case <-ticker.C:
			fm.checkAndFailover()
		}
	}
}

// checkAndFailover checks all shards and triggers failover if needed
func (fm *FailoverManager) checkAndFailover() {
	fm.coordinator.mu.RLock()
	shards := make(map[int][]*ShardNode)
	for shardID, nodes := range fm.coordinator.shards {
		shards[shardID] = append([]*ShardNode{}, nodes...)
	}
	fm.coordinator.mu.RUnlock()

	for shardID, nodes := range shards {
		fm.checkShard(shardID, nodes)
	}
}

// checkShard checks a single shard and triggers failover if primary is unhealthy
func (fm *FailoverManager) checkShard(shardID int, nodes []*ShardNode) {
	fm.mu.Lock()
	state := fm.shardStates[shardID]
	if state == nil {
		state = &shardFailoverState{
			shardID:       shardID,
			lastCheckTime: time.Now(),
		}
		fm.shardStates[shardID] = state
	}
	fm.mu.Unlock()

	// Find primary and replicas
	var primary *ShardNode
	replicas := make([]*ShardNode, 0)

	for _, node := range nodes {
		if node.Role == RolePrimary {
			primary = node
		} else if node.Role == RoleReplica {
			replicas = append(replicas, node)
		}
	}

	// No primary found
	if primary == nil {
		fmt.Printf("⚠️  Shard %d: No primary found\n", shardID)
		return
	}

	// Check primary health
	now := time.Now()
	isPrimaryHealthy := primary.Healthy

	fm.mu.Lock()
	defer fm.mu.Unlock()

	// Primary is healthy - reset unhealthy timer
	if isPrimaryHealthy {
		if state.primaryUnhealthySince != nil {
			fmt.Printf("✅ Shard %d: Primary %s recovered\n", shardID, primary.NodeID)
			state.primaryUnhealthySince = nil
		}
		state.lastCheckTime = now
		return
	}

	// Primary is unhealthy
	if state.primaryUnhealthySince == nil {
		// First time seeing unhealthy
		state.primaryUnhealthySince = &now
		fmt.Printf("⚠️  Shard %d: Primary %s became unhealthy (will failover in %v)\n",
			shardID, primary.NodeID, fm.config.UnhealthyThreshold)
		return
	}

	// Check if unhealthy threshold exceeded
	unhealthyDuration := now.Sub(*state.primaryUnhealthySince)
	if unhealthyDuration < fm.config.UnhealthyThreshold {
		// Not yet time to failover
		remaining := fm.config.UnhealthyThreshold - unhealthyDuration
		fmt.Printf("⏳ Shard %d: Primary unhealthy for %v (failover in %v)\n",
			shardID, unhealthyDuration.Round(time.Second), remaining.Round(time.Second))
		return
	}

	// Failover threshold exceeded - trigger failover
	if state.failoverInProgress {
		// Already failing over
		return
	}

	state.failoverInProgress = true
	fm.mu.Unlock()

	// Perform failover (non-blocking)
	go fm.performFailover(shardID, primary, replicas, state)

	fm.mu.Lock()
}

// performFailover executes the failover process
func (fm *FailoverManager) performFailover(shardID int, oldPrimary *ShardNode, replicas []*ShardNode, state *shardFailoverState) {
	defer func() {
		fm.mu.Lock()
		state.failoverInProgress = false
		fm.mu.Unlock()
	}()

	fmt.Printf("🔄 Shard %d: FAILOVER INITIATED - Primary %s unhealthy\n", shardID, oldPrimary.NodeID)

	// Step 1: Select best replica
	bestReplica := fm.selectBestReplica(replicas)
	if bestReplica == nil {
		fmt.Printf("❌ Shard %d: Failover FAILED - No healthy replicas available\n", shardID)
		return
	}

	fmt.Printf("   → Selected replica %s for promotion (lag: %d ops)\n",
		bestReplica.NodeID, bestReplica.ReplicationLag)

	// Step 2: Promote replica to primary
	if err := fm.promoteReplica(bestReplica); err != nil {
		fmt.Printf("❌ Shard %d: Failover FAILED - Could not promote replica: %v\n", shardID, err)
		return
	}

	// Step 3: Update coordinator routing
	fm.updateCoordinatorRouting(shardID, oldPrimary, bestReplica)

	// Step 4: Mark old primary as replica (or remove if permanently dead)
	fm.demotePrimary(oldPrimary)

	fmt.Printf("✅ Shard %d: FAILOVER COMPLETE - New primary: %s (old: %s)\n",
		shardID, bestReplica.NodeID, oldPrimary.NodeID)

	// Reset state
	fm.mu.Lock()
	state.primaryUnhealthySince = nil
	fm.mu.Unlock()
}

// selectBestReplica selects the best replica for promotion based on health and replication lag
func (fm *FailoverManager) selectBestReplica(replicas []*ShardNode) *ShardNode {
	var best *ShardNode
	minLag := int(1<<31 - 1) // Max int

	for _, replica := range replicas {
		if !replica.Healthy {
			continue
		}

		if replica.ReplicationLag < minLag {
			minLag = replica.ReplicationLag
			best = replica
		}
	}

	return best
}

// promoteReplica promotes a replica to primary
func (fm *FailoverManager) promoteReplica(replica *ShardNode) error {
	// In a real implementation, this would call the replica's /admin/promote endpoint
	// For now, we just update the local state

	fm.coordinator.mu.Lock()
	defer fm.coordinator.mu.Unlock()

	// Find the replica in coordinator's shard list and change its role
	if nodes, ok := fm.coordinator.shards[replica.ShardID]; ok {
		for _, node := range nodes {
			if node.NodeID == replica.NodeID {
				node.Role = RolePrimary
				fmt.Printf("   → Role changed: %s is now PRIMARY\n", node.NodeID)
				return nil
			}
		}
	}

	return fmt.Errorf("replica %s not found in coordinator", replica.NodeID)
}

// updateCoordinatorRouting updates routing tables after failover
func (fm *FailoverManager) updateCoordinatorRouting(shardID int, oldPrimary, newPrimary *ShardNode) {
	// Coordinator automatically uses the primary role when selecting nodes for writes
	// No additional routing update needed since we already changed the role
	fmt.Printf("   → Routing updated: writes now go to %s\n", newPrimary.NodeID)
}

// demotePrimary demotes the old primary to replica or removes it
func (fm *FailoverManager) demotePrimary(oldPrimary *ShardNode) {
	fm.coordinator.mu.Lock()
	defer fm.coordinator.mu.Unlock()

	// Option 1: Mark as replica (if we think it might recover)
	oldPrimary.Role = RoleReplica
	fmt.Printf("   → Old primary %s demoted to REPLICA\n", oldPrimary.NodeID)

	// Option 2: Remove completely (if permanently dead)
	// fm.coordinator.UnregisterShard(oldPrimary.NodeID)
}

// GetShardState returns the current failover state for a shard
func (fm *FailoverManager) GetShardState(shardID int) *shardFailoverState {
	fm.mu.RLock()
	defer fm.mu.RUnlock()
	return fm.shardStates[shardID]
}

// ManualFailover allows manual triggering of failover for a shard
func (fm *FailoverManager) ManualFailover(shardID int) error {
	fm.coordinator.mu.RLock()
	nodes, ok := fm.coordinator.shards[shardID]
	fm.coordinator.mu.RUnlock()

	if !ok {
		return fmt.Errorf("shard %d not found", shardID)
	}

	// Find primary and replicas
	var primary *ShardNode
	replicas := make([]*ShardNode, 0)

	for _, node := range nodes {
		if node.Role == RolePrimary {
			primary = node
		} else if node.Role == RoleReplica {
			replicas = append(replicas, node)
		}
	}

	if primary == nil {
		return fmt.Errorf("no primary found for shard %d", shardID)
	}

	if len(replicas) == 0 {
		return fmt.Errorf("no replicas available for shard %d", shardID)
	}

	fm.mu.Lock()
	state := fm.shardStates[shardID]
	if state == nil {
		state = &shardFailoverState{shardID: shardID}
		fm.shardStates[shardID] = state
	}
	fm.mu.Unlock()

	fmt.Printf("🔧 Manual failover triggered for shard %d\n", shardID)
	go fm.performFailover(shardID, primary, replicas, state)

	return nil
}

// GetFailoverStats returns statistics about failover activity
func (fm *FailoverManager) GetFailoverStats() map[string]any {
	fm.mu.RLock()
	defer fm.mu.RUnlock()

	stats := map[string]any{
		"enabled":             fm.config.EnableAutoFailover,
		"unhealthy_threshold": fm.config.UnhealthyThreshold.String(),
		"check_interval":      fm.config.CheckInterval.String(),
		"shards":              make([]map[string]any, 0),
	}

	for shardID, state := range fm.shardStates {
		shardInfo := map[string]any{
			"shard_id":            shardID,
			"failover_in_progress": state.failoverInProgress,
			"last_check":          state.lastCheckTime.Unix(),
		}

		if state.primaryUnhealthySince != nil {
			shardInfo["primary_unhealthy_since"] = state.primaryUnhealthySince.Unix()
			shardInfo["unhealthy_duration_seconds"] = time.Since(*state.primaryUnhealthySince).Seconds()
		}

		stats["shards"] = append(stats["shards"].([]map[string]any), shardInfo)
	}

	return stats
}
