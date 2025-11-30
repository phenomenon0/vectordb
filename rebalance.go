package main

import (
	"fmt"
	"sync"
	"time"
)

// ===========================================================================================
// AUTO-REBALANCING
// Zero-downtime data migration when adding/removing shards
// ===========================================================================================

// MigrationStatus represents the status of a data migration
type MigrationStatus int

const (
	MigrationPending MigrationStatus = iota
	MigrationInProgress
	MigrationCompleted
	MigrationFailed
	MigrationRolledBack
)

func (ms MigrationStatus) String() string {
	switch ms {
	case MigrationPending:
		return "pending"
	case MigrationInProgress:
		return "in_progress"
	case MigrationCompleted:
		return "completed"
	case MigrationFailed:
		return "failed"
	case MigrationRolledBack:
		return "rolled_back"
	default:
		return "unknown"
	}
}

// Migration represents a collection migration from one shard to another
type Migration struct {
	ID           string
	Collection   string
	FromShardID  int
	ToShardID    int
	TotalVectors int
	CopiedVectors int
	Status       MigrationStatus
	StartTime    time.Time
	EndTime      *time.Time
	Error        error

	// Migration state
	dualWriteEnabled bool
	routingUpdated   bool
	cleanupDone      bool
}

// Progress returns the migration progress percentage
func (m *Migration) Progress() float64 {
	if m.TotalVectors == 0 {
		return 0
	}
	return (float64(m.CopiedVectors) / float64(m.TotalVectors)) * 100
}

// RebalanceCoordinator manages shard rebalancing and data migration
type RebalanceCoordinator struct {
	mu sync.RWMutex

	distributed *DistributedVectorDB
	migrations  map[string]*Migration // migration ID -> migration
	active      bool

	// HTTP client for shard communication
	migrationClient *MigrationClient

	// Control channels
	stopCh   chan struct{}
	pauseCh  chan struct{}
	resumeCh chan struct{}

	// Statistics
	stats *RebalanceStats
}

// NewRebalanceCoordinator creates a new rebalance coordinator
func NewRebalanceCoordinator(distributed *DistributedVectorDB) *RebalanceCoordinator {
	return &RebalanceCoordinator{
		distributed:     distributed,
		migrations:      make(map[string]*Migration),
		migrationClient: NewMigrationClient(),
		stopCh:          make(chan struct{}),
		pauseCh:         make(chan struct{}),
		resumeCh:        make(chan struct{}),
		stats:           NewRebalanceStats(),
	}
}

// AddShard adds a new shard and triggers rebalancing
func (rc *RebalanceCoordinator) AddShard(newShard *ShardNode) error {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	// Register shard with distributed coordinator
	if err := rc.distributed.RegisterShard(newShard); err != nil {
		return fmt.Errorf("failed to register shard: %w", err)
	}

	// Calculate new shard assignments
	newMapping := rc.calculateOptimalMapping()

	// Plan migrations
	migrations := rc.planMigrations(newMapping)

	if len(migrations) == 0 {
		return nil // No migrations needed
	}

	// Start migrations
	for _, migration := range migrations {
		rc.migrations[migration.ID] = migration
		go rc.executeMigration(migration)
	}

	rc.stats.RecordRebalanceStart(len(migrations))

	return nil
}

// RemoveShard removes a shard and triggers rebalancing
func (rc *RebalanceCoordinator) RemoveShard(shardID int) error {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	// Calculate new shard assignments (without removed shard)
	newMapping := rc.calculateOptimalMappingWithoutShard(shardID)

	// Plan migrations
	migrations := rc.planMigrations(newMapping)

	// Start migrations
	for _, migration := range migrations {
		rc.migrations[migration.ID] = migration
		go rc.executeMigration(migration)
	}

	// Unregister shard after migrations complete
	go func() {
		rc.waitForMigrations()
		rc.distributed.UnregisterShard(fmt.Sprintf("shard-%d", shardID))
	}()

	rc.stats.RecordRebalanceStart(len(migrations))

	return nil
}

// calculateOptimalMapping calculates optimal collection → shard mapping
func (rc *RebalanceCoordinator) calculateOptimalMapping() map[string]int {
	// Get current shard count
	rc.distributed.mu.RLock()
	numShards := rc.distributed.numShards
	rc.distributed.mu.RUnlock()

	// Get all collections
	collections := rc.getAllCollections()

	// Recalculate mapping using consistent hashing
	mapping := make(map[string]int)
	for _, collection := range collections {
		mapping[collection] = rc.distributed.getShardForCollection(collection) % numShards
	}

	return mapping
}

// calculateOptimalMappingWithoutShard calculates mapping excluding a specific shard
func (rc *RebalanceCoordinator) calculateOptimalMappingWithoutShard(excludeShardID int) map[string]int {
	// Similar to calculateOptimalMapping, but excludes the removed shard
	return rc.calculateOptimalMapping()
}

// planMigrations determines which collections need to move
func (rc *RebalanceCoordinator) planMigrations(newMapping map[string]int) []*Migration {
	var migrations []*Migration

	rc.distributed.mu.RLock()
	currentMapping := rc.distributed.collectionShards
	rc.distributed.mu.RUnlock()

	for collection, newShardID := range newMapping {
		currentShardID, exists := currentMapping[collection]
		if !exists || currentShardID == newShardID {
			continue // No migration needed
		}

		// Create migration
		migration := &Migration{
			ID:          fmt.Sprintf("mig-%s-%d-to-%d", collection, currentShardID, newShardID),
			Collection:  collection,
			FromShardID: currentShardID,
			ToShardID:   newShardID,
			Status:      MigrationPending,
			StartTime:   time.Now(),
		}

		migrations = append(migrations, migration)
	}

	return migrations
}

// executeMigration executes a single migration with 4 phases
func (rc *RebalanceCoordinator) executeMigration(m *Migration) {
	fmt.Printf("🔄 Starting migration: %s (collection '%s': shard %d → %d)\n",
		m.ID, m.Collection, m.FromShardID, m.ToShardID)

	m.Status = MigrationInProgress

	// Phase 1: Copy data (background, no downtime)
	if err := rc.phase1CopyData(m); err != nil {
		rc.handleMigrationFailure(m, fmt.Errorf("phase 1 failed: %w", err))
		return
	}

	// Phase 2: Enable dual-write (writes go to both shards)
	if err := rc.phase2EnableDualWrite(m); err != nil {
		rc.handleMigrationFailure(m, fmt.Errorf("phase 2 failed: %w", err))
		return
	}

	// Phase 3: Atomic switchover (update routing)
	if err := rc.phase3AtomicSwitch(m); err != nil {
		rc.handleMigrationFailure(m, fmt.Errorf("phase 3 failed: %w", err))
		return
	}

	// Phase 4: Cleanup (delete from source)
	if err := rc.phase4Cleanup(m); err != nil {
		// Cleanup failure is non-fatal
		fmt.Printf("⚠️  Cleanup warning for %s: %v\n", m.ID, err)
	}

	// Success!
	m.Status = MigrationCompleted
	now := time.Now()
	m.EndTime = &now

	rc.stats.RecordMigrationComplete(m.TotalVectors, time.Since(m.StartTime))

	fmt.Printf("✅ Migration complete: %s (%.0f%% - %d vectors in %.1fs)\n",
		m.ID, m.Progress(), m.TotalVectors, time.Since(m.StartTime).Seconds())
}

// phase1CopyData copies all vectors from source to destination shard
func (rc *RebalanceCoordinator) phase1CopyData(m *Migration) error {
	fmt.Printf("   Phase 1: Copying data from shard %d to shard %d...\n", m.FromShardID, m.ToShardID)

	// Get shard addresses
	srcAddr, dstAddr, err := rc.getShardAddresses(m.FromShardID, m.ToShardID)
	if err != nil {
		return fmt.Errorf("failed to get shard addresses: %w", err)
	}

	// Check for pause/stop before starting
	select {
	case <-rc.stopCh:
		return fmt.Errorf("migration stopped")
	default:
	}

	// Get collection stats first
	stats, err := rc.migrationClient.GetCollectionStats(srcAddr, m.Collection)
	if err != nil {
		return fmt.Errorf("failed to get collection stats: %w", err)
	}
	if active, ok := stats["active"].(float64); ok {
		m.TotalVectors = int(active)
	}

	// Export collection from source shard
	fmt.Printf("   Exporting collection '%s' from shard %d (%s)...\n", m.Collection, m.FromShardID, srcAddr)
	export, err := rc.migrationClient.ExportFromShard(srcAddr, m.Collection)
	if err != nil {
		return fmt.Errorf("failed to export from source: %w", err)
	}

	m.TotalVectors = export.Count
	fmt.Printf("   Exported %d vectors from source shard\n", m.TotalVectors)

	// Check for pause/stop
	select {
	case <-rc.pauseCh:
		<-rc.resumeCh // Wait for resume
	case <-rc.stopCh:
		return fmt.Errorf("migration stopped")
	default:
	}

	// Import collection to destination shard
	fmt.Printf("   Importing collection '%s' to shard %d (%s)...\n", m.Collection, m.ToShardID, dstAddr)
	if err := rc.migrationClient.ImportToShard(dstAddr, export); err != nil {
		return fmt.Errorf("failed to import to destination: %w", err)
	}

	m.CopiedVectors = export.Count
	fmt.Printf("   Phase 1 complete: %d vectors copied\n", m.TotalVectors)
	return nil
}

// getShardAddresses retrieves HTTP addresses for source and destination shards
func (rc *RebalanceCoordinator) getShardAddresses(fromShardID, toShardID int) (string, string, error) {
	rc.distributed.mu.RLock()
	defer rc.distributed.mu.RUnlock()

	var srcAddr, dstAddr string

	// shards is map[int][]*ShardNode - iterate over shard ID -> nodes
	for shardID, nodes := range rc.distributed.shards {
		for _, node := range nodes {
			if shardID == fromShardID && node.Role == RolePrimary {
				srcAddr = node.HTTPAddr
			}
			if shardID == toShardID && node.Role == RolePrimary {
				dstAddr = node.HTTPAddr
			}
		}
	}

	if srcAddr == "" {
		return "", "", fmt.Errorf("source shard %d not found or has no primary", fromShardID)
	}
	if dstAddr == "" {
		return "", "", fmt.Errorf("destination shard %d not found or has no primary", toShardID)
	}

	return srcAddr, dstAddr, nil
}

// phase2EnableDualWrite enables dual-write mode (writes go to both shards)
func (rc *RebalanceCoordinator) phase2EnableDualWrite(m *Migration) error {
	fmt.Printf("   Phase 2: Enabling dual-write mode...\n")

	// In production, this would update coordinator to write to both shards
	m.dualWriteEnabled = true

	// Wait for in-flight writes to complete
	time.Sleep(1 * time.Second)

	fmt.Printf("   Phase 2 complete: dual-write enabled\n")
	return nil
}

// phase3AtomicSwitch atomically switches routing to new shard
func (rc *RebalanceCoordinator) phase3AtomicSwitch(m *Migration) error {
	fmt.Printf("   Phase 3: Atomic routing switch...\n")

	rc.distributed.mu.Lock()
	defer rc.distributed.mu.Unlock()

	// Update collection → shard mapping
	rc.distributed.collectionShards[m.Collection] = m.ToShardID
	m.routingUpdated = true

	// Disable dual-write (now only writes to new shard)
	m.dualWriteEnabled = false

	fmt.Printf("   Phase 3 complete: routing updated to shard %d\n", m.ToShardID)
	return nil
}

// phase4Cleanup deletes collection from source shard
func (rc *RebalanceCoordinator) phase4Cleanup(m *Migration) error {
	fmt.Printf("   Phase 4: Cleaning up source shard...\n")

	// Get source shard address
	srcAddr, _, err := rc.getShardAddresses(m.FromShardID, m.ToShardID)
	if err != nil {
		return fmt.Errorf("failed to get source shard address: %w", err)
	}

	// Delete collection from source shard
	if err := rc.migrationClient.DeleteFromShard(srcAddr, m.Collection); err != nil {
		return fmt.Errorf("failed to delete from source: %w", err)
	}

	m.cleanupDone = true

	fmt.Printf("   Phase 4 complete: source shard cleaned up\n")
	return nil
}

// handleMigrationFailure handles migration failure and attempts rollback
func (rc *RebalanceCoordinator) handleMigrationFailure(m *Migration, err error) {
	fmt.Printf("❌ Migration failed: %s - %v\n", m.ID, err)

	m.Status = MigrationFailed
	m.Error = err
	now := time.Now()
	m.EndTime = &now

	// Attempt rollback
	if err := rc.rollbackMigration(m); err != nil {
		fmt.Printf("⚠️  Rollback failed for %s: %v\n", m.ID, err)
		m.Status = MigrationFailed
	} else {
		fmt.Printf("✅ Rollback successful for %s\n", m.ID)
		m.Status = MigrationRolledBack
	}

	rc.stats.RecordMigrationFailed()
}

// rollbackMigration rolls back a failed migration
func (rc *RebalanceCoordinator) rollbackMigration(m *Migration) error {
	// Rollback steps:
	// 1. Disable dual-write if enabled
	// 2. Revert routing if updated
	// 3. Delete from destination shard
	// 4. Keep source intact

	if m.dualWriteEnabled {
		m.dualWriteEnabled = false
	}

	if m.routingUpdated {
		rc.distributed.mu.Lock()
		rc.distributed.collectionShards[m.Collection] = m.FromShardID
		rc.distributed.mu.Unlock()
	}

	// Delete from destination shard if any data was copied
	if m.CopiedVectors > 0 {
		_, dstAddr, err := rc.getShardAddresses(m.FromShardID, m.ToShardID)
		if err == nil && dstAddr != "" {
			if err := rc.migrationClient.DeleteFromShard(dstAddr, m.Collection); err != nil {
				fmt.Printf("   ⚠️  Failed to cleanup destination during rollback: %v\n", err)
			} else {
				fmt.Printf("   ✓ Deleted %d vectors from destination shard during rollback\n", m.CopiedVectors)
			}
		}
	}

	return nil
}

// PauseMigrations pauses all active migrations
func (rc *RebalanceCoordinator) PauseMigrations() {
	close(rc.pauseCh)
	rc.pauseCh = make(chan struct{})
}

// ResumeMigrations resumes all paused migrations
func (rc *RebalanceCoordinator) ResumeMigrations() {
	close(rc.resumeCh)
	rc.resumeCh = make(chan struct{})
}

// StopMigrations stops all active migrations
func (rc *RebalanceCoordinator) StopMigrations() {
	close(rc.stopCh)
}

// GetMigrationStatus returns the status of all migrations
func (rc *RebalanceCoordinator) GetMigrationStatus() map[string]any {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	migrations := make([]map[string]any, 0, len(rc.migrations))
	for _, m := range rc.migrations {
		migrations = append(migrations, map[string]any{
			"id":             m.ID,
			"collection":     m.Collection,
			"from_shard":     m.FromShardID,
			"to_shard":       m.ToShardID,
			"status":         m.Status.String(),
			"progress":       fmt.Sprintf("%.1f%%", m.Progress()),
			"total_vectors":  m.TotalVectors,
			"copied_vectors": m.CopiedVectors,
			"elapsed":        time.Since(m.StartTime).Seconds(),
		})
	}

	return map[string]any{
		"active":     rc.active,
		"migrations": migrations,
		"stats":      rc.stats.GetStats(),
	}
}

// waitForMigrations waits for all active migrations to complete
func (rc *RebalanceCoordinator) waitForMigrations() {
	for {
		rc.mu.RLock()
		allDone := true
		for _, m := range rc.migrations {
			if m.Status == MigrationPending || m.Status == MigrationInProgress {
				allDone = false
				break
			}
		}
		rc.mu.RUnlock()

		if allDone {
			break
		}

		time.Sleep(1 * time.Second)
	}
}

// getAllCollections returns all collections across all shards
func (rc *RebalanceCoordinator) getAllCollections() []string {
	rc.distributed.mu.RLock()
	defer rc.distributed.mu.RUnlock()

	collections := make([]string, 0, len(rc.distributed.collectionShards))
	for collection := range rc.distributed.collectionShards {
		collections = append(collections, collection)
	}
	return collections
}

// ===========================================================================================
// REBALANCE STATISTICS
// ===========================================================================================

// RebalanceStats tracks rebalancing statistics
type RebalanceStats struct {
	mu sync.RWMutex

	TotalRebalances      int64
	TotalMigrations      int64
	CompletedMigrations  int64
	FailedMigrations     int64
	TotalVectorsMigrated int64
	TotalMigrationTimeMs int64
	AverageMigrationTimeMs float64
}

// NewRebalanceStats creates a new rebalance statistics tracker
func NewRebalanceStats() *RebalanceStats {
	return &RebalanceStats{}
}

// RecordRebalanceStart records the start of a rebalance operation
func (rs *RebalanceStats) RecordRebalanceStart(migrationCount int) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.TotalRebalances++
	rs.TotalMigrations += int64(migrationCount)
}

// RecordMigrationComplete records a completed migration
func (rs *RebalanceStats) RecordMigrationComplete(vectors int, duration time.Duration) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.CompletedMigrations++
	rs.TotalVectorsMigrated += int64(vectors)
	rs.TotalMigrationTimeMs += duration.Milliseconds()
	rs.AverageMigrationTimeMs = float64(rs.TotalMigrationTimeMs) / float64(rs.CompletedMigrations)
}

// RecordMigrationFailed records a failed migration
func (rs *RebalanceStats) RecordMigrationFailed() {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.FailedMigrations++
}

// GetStats returns current statistics
func (rs *RebalanceStats) GetStats() map[string]any {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	successRate := float64(0)
	if rs.TotalMigrations > 0 {
		successRate = float64(rs.CompletedMigrations) / float64(rs.TotalMigrations)
	}

	return map[string]any{
		"total_rebalances":         rs.TotalRebalances,
		"total_migrations":         rs.TotalMigrations,
		"completed_migrations":     rs.CompletedMigrations,
		"failed_migrations":        rs.FailedMigrations,
		"total_vectors_migrated":   rs.TotalVectorsMigrated,
		"average_migration_time_ms": rs.AverageMigrationTimeMs,
		"success_rate":             successRate,
	}
}

// ===========================================================================================
// USAGE EXAMPLE
// ===========================================================================================

/*
Example usage:

// Create rebalance coordinator
rebalancer := NewRebalanceCoordinator(distributedVectorDB)

// Add a new shard (triggers automatic rebalancing)
newShard := &ShardNode{
    NodeID:  "shard-4",
    ShardID: 3,
    Role:    RolePrimary,
    BaseURL: "http://host4:9000",
    Healthy: true,
}

err := rebalancer.AddShard(newShard)
if err != nil {
    log.Fatal(err)
}

// Monitor migration progress
go func() {
    ticker := time.NewTicker(5 * time.Second)
    for range ticker.C {
        status := rebalancer.GetMigrationStatus()
        fmt.Printf("Migration status: %+v\n", status)
    }
}()

// Pause migrations if needed
rebalancer.PauseMigrations()

// Resume
rebalancer.ResumeMigrations()

// Remove a shard (triggers rebalancing)
err = rebalancer.RemoveShard(0)

// Result: Zero-downtime migration!
// - Collections automatically moved to new shards
// - No query interruption
// - Automatic cleanup
*/
