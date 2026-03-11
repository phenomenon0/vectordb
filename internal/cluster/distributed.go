package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"net/http"
	"sort"
	"sync"
	"time"
)

// ===========================================================================================
// DISTRIBUTED VECTORDB COORDINATOR
// Handles sharding, replication, and query routing across multiple vectordb instances
// ===========================================================================================

// ReplicaRole defines the role of a vectordb node in a shard
type ReplicaRole string

const (
	RolePrimary ReplicaRole = "primary"
	RoleReplica ReplicaRole = "replica"
)

// ReadStrategy defines how to route read queries
type ReadStrategy string

const (
	ReadPrimaryOnly   ReadStrategy = "primary-only"   // All reads to primary (strong consistency)
	ReadReplicaPrefer ReadStrategy = "replica-prefer" // Prefer replicas (eventual consistency)
	ReadBalanced      ReadStrategy = "balanced"       // Load balance across all (primary + replicas)
)

// ShardingStrategy defines how to assign data to shards
type ShardingStrategy string

const (
	// ShardByCollection routes all vectors in a collection to the same shard (default).
	ShardByCollection ShardingStrategy = "collection"
	// ShardByVector routes each vector to a shard based on its ID.
	// Queries scatter to all shards and merge top-K results.
	ShardByVector ShardingStrategy = "vector"
)

// ShardNode represents a single vectordb instance in a shard
type ShardNode struct {
	NodeID         string      `json:"node_id"`
	ShardID        int         `json:"shard_id"`
	Role           ReplicaRole `json:"role"`
	HTTPAddr       string      `json:"http_addr"`
	Healthy        bool        `json:"healthy"`
	LastSeen       time.Time   `json:"last_seen"`
	ReplicationLag int         `json:"replication_lag"`
}

// DistributedVectorDB coordinates multiple vectordb shards
// maxCollectionCacheSize limits the collection shard cache to prevent unbounded memory growth
const maxCollectionCacheSize = 10000

type DistributedVectorDB struct {
	mu sync.RWMutex

	numShards         int
	replicationFactor int
	readStrategy      ReadStrategy
	shardingStrategy  ShardingStrategy

	// Shard topology: shardID -> nodes (primary + replicas)
	shards map[int][]*ShardNode

	// Collection -> ShardID mapping cache (limited to maxCollectionCacheSize entries)
	collectionShards map[string]int

	// HTTP client for inter-shard communication
	httpClient *http.Client

	// Health monitoring
	healthCheckInterval time.Duration
	stopHealthCheck     chan struct{}

	// Quorum-based consensus for distributed decisions
	quorum  *QuorumVoter
	fencing *FencingManager
}

// DistributedConfig configures the distributed vectordb
type DistributedConfig struct {
	NumShards           int
	ReplicationFactor   int              // Number of replicas per shard
	ReadStrategy        ReadStrategy     // How to route read queries
	ShardingStrategy    ShardingStrategy // How to assign data to shards (default: "collection")
	HealthCheckInterval time.Duration

	// Quorum configuration
	CoordinatorID    string   // This coordinator's unique ID
	PeerCoordinators []string // HTTP addresses of other coordinators for voting
}

// NewDistributedVectorDB creates a new distributed vectordb coordinator.
//
// WARNING: Distributed mode is EXPERIMENTAL and not recommended for production use.
// Known limitations:
//   - Quorum safety checks are incomplete (see quorum.go TODOs)
//   - Snapshot sync for far-behind replicas is not implemented
//   - Leader election edge cases are not fully handled
//
// For production deployments, use single-node mode until distributed mode reaches v1.0.
func NewDistributedVectorDB(cfg DistributedConfig) *DistributedVectorDB {
	fmt.Println("WARNING: Distributed mode is EXPERIMENTAL - not recommended for production")
	fmt.Println("   Known issues: incomplete quorum checks, limited snapshot sync")
	fmt.Println("   For production, use single-node mode")

	if cfg.HealthCheckInterval == 0 {
		cfg.HealthCheckInterval = 10 * time.Second
	}
	if cfg.ReadStrategy == "" {
		cfg.ReadStrategy = ReadReplicaPrefer
	}
	if cfg.ShardingStrategy == "" {
		cfg.ShardingStrategy = ShardByCollection
	}
	if cfg.CoordinatorID == "" {
		cfg.CoordinatorID = fmt.Sprintf("coordinator-%d", time.Now().Unix())
	}

	d := &DistributedVectorDB{
		numShards:           cfg.NumShards,
		replicationFactor:   cfg.ReplicationFactor,
		readStrategy:        cfg.ReadStrategy,
		shardingStrategy:    cfg.ShardingStrategy,
		shards:              make(map[int][]*ShardNode),
		collectionShards:    make(map[string]int),
		httpClient:          &http.Client{Timeout: 30 * time.Second},
		healthCheckInterval: cfg.HealthCheckInterval,
		stopHealthCheck:     make(chan struct{}),
	}

	// Initialize quorum voting if peers configured
	if len(cfg.PeerCoordinators) > 0 {
		d.quorum = NewQuorumVoter(cfg.CoordinatorID, cfg.PeerCoordinators)
		d.fencing = NewFencingManager(d.quorum)
		fmt.Printf("Quorum voting enabled: coordinator=%s, peers=%d\n",
			cfg.CoordinatorID, len(cfg.PeerCoordinators))
	} else {
		fmt.Printf("Running in single-coordinator mode (no quorum)\n")
	}

	// Start health monitoring
	go d.healthCheckLoop()

	return d
}

// RegisterShard adds a shard node to the cluster
func (d *DistributedVectorDB) RegisterShard(node *ShardNode) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if node.ShardID < 0 || node.ShardID >= d.numShards {
		return fmt.Errorf("invalid shard ID %d (must be 0-%d)", node.ShardID, d.numShards-1)
	}

	// Initialize shard if not exists
	if d.shards[node.ShardID] == nil {
		d.shards[node.ShardID] = make([]*ShardNode, 0)
	}

	// Check for duplicate node
	for _, existing := range d.shards[node.ShardID] {
		if existing.NodeID == node.NodeID {
			return fmt.Errorf("node %s already registered", node.NodeID)
		}
	}

	// Validate primary/replica constraints
	primaryCount := 0
	for _, n := range d.shards[node.ShardID] {
		if n.Role == RolePrimary {
			primaryCount++
		}
	}

	if node.Role == RolePrimary && primaryCount > 0 {
		return fmt.Errorf("shard %d already has a primary", node.ShardID)
	}

	node.Healthy = true
	node.LastSeen = time.Now()
	d.shards[node.ShardID] = append(d.shards[node.ShardID], node)

	return nil
}

// UnregisterShard removes a shard node from the cluster
func (d *DistributedVectorDB) UnregisterShard(nodeID string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	for shardID, nodes := range d.shards {
		for i, node := range nodes {
			if node.NodeID == nodeID {
				// Remove node from list
				d.shards[shardID] = append(nodes[:i], nodes[i+1:]...)
				return nil
			}
		}
	}

	return fmt.Errorf("node %s not found", nodeID)
}

// getShardForCollection returns the shard ID for a given collection using consistent hashing
func (d *DistributedVectorDB) getShardForCollection(collection string) int {
	d.mu.RLock()
	// Check cache first
	if shardID, ok := d.collectionShards[collection]; ok {
		d.mu.RUnlock()
		return shardID
	}
	d.mu.RUnlock()

	// Compute shard using consistent hashing
	h := fnv.New64a()
	h.Write([]byte(collection))
	shardID := int(h.Sum64() % uint64(d.numShards))

	// Cache the result (with size limit to prevent unbounded growth)
	d.mu.Lock()
	if len(d.collectionShards) >= maxCollectionCacheSize {
		// Evict random entries when cache is full (simple eviction strategy)
		count := 0
		for k := range d.collectionShards {
			delete(d.collectionShards, k)
			count++
			if count >= maxCollectionCacheSize/10 { // Evict 10%
				break
			}
		}
	}
	d.collectionShards[collection] = shardID
	d.mu.Unlock()

	return shardID
}

// getShardForVector returns the shard ID for a vector ID using FNV-64a hashing.
// Used when sharding strategy is "vector".
func (d *DistributedVectorDB) getShardForVector(vectorID string) int {
	h := fnv.New64a()
	h.Write([]byte(vectorID))
	return int(h.Sum64() % uint64(d.numShards))
}

// getShardsForInsert returns the shard(s) that should receive an insert.
// For collection sharding: returns a single shard based on collection name.
// For vector sharding: returns a single shard based on vector ID.
func (d *DistributedVectorDB) getShardsForInsert(collection string, vectorID string) []int {
	if d.shardingStrategy == ShardByVector {
		return []int{d.getShardForVector(vectorID)}
	}
	return []int{d.getShardForCollection(collection)}
}

// getShardsForQuery returns the shard(s) to query.
// For collection sharding: returns shards for the requested collections.
// For vector sharding: returns ALL shards (scatter-gather).
func (d *DistributedVectorDB) getShardsForQuery(collections []string) map[int]bool {
	shardsToQuery := make(map[int]bool)

	if d.shardingStrategy == ShardByVector {
		// Vector sharding: must scatter to all shards
		d.mu.RLock()
		for shardID := range d.shards {
			shardsToQuery[shardID] = true
		}
		d.mu.RUnlock()
		return shardsToQuery
	}

	// Collection sharding: route to specific shards
	if len(collections) == 0 {
		d.mu.RLock()
		for shardID := range d.shards {
			shardsToQuery[shardID] = true
		}
		d.mu.RUnlock()
	} else {
		for _, collection := range collections {
			shardID := d.getShardForCollection(collection)
			shardsToQuery[shardID] = true
		}
	}
	return shardsToQuery
}

// getShardsForDelete returns the shard(s) that should receive a delete.
// For collection sharding: returns a single shard based on collection name.
// For vector sharding: returns a single shard based on vector ID.
func (d *DistributedVectorDB) getShardsForDelete(collection string, vectorID string) []int {
	if d.shardingStrategy == ShardByVector {
		return []int{d.getShardForVector(vectorID)}
	}
	return []int{d.getShardForCollection(collection)}
}

// InvalidateCollectionCache removes a collection from the shard cache.
// Call this when a collection is deleted or when shard topology changes.
func (d *DistributedVectorDB) InvalidateCollectionCache(collection string) {
	d.mu.Lock()
	delete(d.collectionShards, collection)
	d.mu.Unlock()
}

// ClearCollectionCache removes all entries from the collection shard cache.
// Call this when numShards changes or during cluster reconfiguration.
func (d *DistributedVectorDB) ClearCollectionCache() {
	d.mu.Lock()
	d.collectionShards = make(map[string]int)
	d.mu.Unlock()
}

// selectNodeForWrite returns the primary node for a given shard
func (d *DistributedVectorDB) selectNodeForWrite(shardID int) (*ShardNode, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	nodes := d.shards[shardID]
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes available for shard %d", shardID)
	}

	// Find healthy primary
	for _, node := range nodes {
		if node.Role == RolePrimary && node.Healthy {
			return node, nil
		}
	}

	return nil, fmt.Errorf("no healthy primary found for shard %d", shardID)
}

// selectNodeForRead returns a node for reading based on the read strategy
func (d *DistributedVectorDB) selectNodeForRead(shardID int) (*ShardNode, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	nodes := d.shards[shardID]
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes available for shard %d", shardID)
	}

	healthyNodes := make([]*ShardNode, 0)
	var primary *ShardNode

	for _, node := range nodes {
		if node.Healthy {
			healthyNodes = append(healthyNodes, node)
			if node.Role == RolePrimary {
				primary = node
			}
		}
	}

	if len(healthyNodes) == 0 {
		return nil, fmt.Errorf("no healthy nodes for shard %d", shardID)
	}

	switch d.readStrategy {
	case ReadPrimaryOnly:
		if primary != nil {
			return primary, nil
		}
		return nil, fmt.Errorf("no healthy primary for shard %d", shardID)

	case ReadReplicaPrefer:
		// Prefer replicas, fallback to primary
		replicas := make([]*ShardNode, 0)
		for _, node := range healthyNodes {
			if node.Role == RoleReplica {
				replicas = append(replicas, node)
			}
		}
		if len(replicas) > 0 {
			// Round-robin or random selection
			return replicas[time.Now().UnixNano()%int64(len(replicas))], nil
		}
		if primary != nil {
			return primary, nil
		}
		return nil, fmt.Errorf("no healthy nodes for shard %d", shardID)

	case ReadBalanced:
		// Round-robin across all healthy nodes
		return healthyNodes[time.Now().UnixNano()%int64(len(healthyNodes))], nil

	default:
		return healthyNodes[0], nil
	}
}

// Add inserts a vector into the distributed vectordb
func (d *DistributedVectorDB) Add(doc string, id string, meta map[string]string, collection string) (string, error) {
	// Route to appropriate shard
	shards := d.getShardsForInsert(collection, id)
	shardID := shards[0]

	// Get primary node for write
	node, err := d.selectNodeForWrite(shardID)
	if err != nil {
		return "", fmt.Errorf("failed to select node: %w", err)
	}

	// Forward request to shard
	reqBody := map[string]any{
		"doc":        doc,
		"id":         id,
		"meta":       meta,
		"collection": collection,
	}
	body, _ := json.Marshal(reqBody)

	resp, err := d.httpClient.Post(node.HTTPAddr+"/insert", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to insert on shard %d: %w", shardID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("insert failed with status %d", resp.StatusCode)
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return result["id"].(string), nil
}

// Query searches across collections
func (d *DistributedVectorDB) Query(query string, topK int, collections []string, filters map[string]string, mode string) ([]map[string]any, error) {
	// Determine which shards to query
	shardsToQuery := d.getShardsForQuery(collections)

	// Query shards in parallel
	type shardResult struct {
		shardID int
		results []map[string]any
		err     error
	}

	resultChan := make(chan shardResult, len(shardsToQuery))
	var wg sync.WaitGroup

	for shardID := range shardsToQuery {
		wg.Add(1)
		go func(sid int) {
			defer wg.Done()

			node, err := d.selectNodeForRead(sid)
			if err != nil {
				resultChan <- shardResult{shardID: sid, err: err}
				return
			}

			// Build query request
			reqBody := map[string]any{
				"query":       query,
				"top_k":       topK,
				"collections": collections,
				"mode":        mode,
			}
			if len(filters) > 0 {
				reqBody["meta_filter"] = filters
			}
			body, _ := json.Marshal(reqBody)

			resp, err := d.httpClient.Post(node.HTTPAddr+"/query", "application/json", bytes.NewReader(body))
			if err != nil {
				resultChan <- shardResult{shardID: sid, err: err}
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				resultChan <- shardResult{shardID: sid, err: fmt.Errorf("query failed with status %d", resp.StatusCode)}
				return
			}

			var queryResp struct {
				IDs    []string            `json:"ids"`
				Docs   []string            `json:"docs"`
				Scores []float64           `json:"scores"`
				Meta   []map[string]string `json:"meta"`
			}

			if err := json.NewDecoder(resp.Body).Decode(&queryResp); err != nil {
				resultChan <- shardResult{shardID: sid, err: err}
				return
			}

			// Convert to result format
			results := make([]map[string]any, len(queryResp.IDs))
			for i := range queryResp.IDs {
				results[i] = map[string]any{
					"id":    queryResp.IDs[i],
					"doc":   queryResp.Docs[i],
					"score": queryResp.Scores[i],
				}
				if i < len(queryResp.Meta) {
					results[i]["meta"] = queryResp.Meta[i]
				}
			}

			resultChan <- shardResult{shardID: sid, results: results}
		}(shardID)
	}

	// Wait for all queries to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Aggregate results
	allResults := make([]map[string]any, 0)
	var errors []error

	for result := range resultChan {
		if result.err != nil {
			errors = append(errors, fmt.Errorf("shard %d: %w", result.shardID, result.err))
			continue
		}
		allResults = append(allResults, result.results...)
	}

	if len(errors) > 0 && len(allResults) == 0 {
		return nil, fmt.Errorf("all shards failed: %v", errors)
	}

	// Sort by score descending
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i]["score"].(float64) > allResults[j]["score"].(float64)
	})

	// Return top-K
	if len(allResults) > topK {
		allResults = allResults[:topK]
	}

	return allResults, nil
}

// Delete removes a vector from a collection
func (d *DistributedVectorDB) Delete(id string, collection string) error {
	// Route to appropriate shard
	shards := d.getShardsForDelete(collection, id)
	shardID := shards[0]

	// Get primary node for write
	node, err := d.selectNodeForWrite(shardID)
	if err != nil {
		return fmt.Errorf("failed to select node: %w", err)
	}

	// Forward request to shard
	reqBody := map[string]any{"id": id}
	body, _ := json.Marshal(reqBody)

	resp, err := d.httpClient.Post(node.HTTPAddr+"/delete", "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to delete on shard %d: %w", shardID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("delete failed with status %d", resp.StatusCode)
	}

	return nil
}

// GetClusterStatus returns the status of all shards
func (d *DistributedVectorDB) GetClusterStatus() map[string]any {
	d.mu.RLock()
	defer d.mu.RUnlock()

	shardStatus := make([]map[string]any, 0)

	for shardID, nodes := range d.shards {
		status := map[string]any{
			"shard_id": shardID,
			"nodes":    make([]map[string]any, 0),
		}

		for _, node := range nodes {
			nodeStatus := map[string]any{
				"node_id":         node.NodeID,
				"role":            node.Role,
				"addr":            node.HTTPAddr,
				"healthy":         node.Healthy,
				"last_seen":       node.LastSeen.Unix(),
				"replication_lag": node.ReplicationLag,
			}
			status["nodes"] = append(status["nodes"].([]map[string]any), nodeStatus)
		}

		shardStatus = append(shardStatus, status)
	}

	return map[string]any{
		"num_shards":         d.numShards,
		"replication_factor": d.replicationFactor,
		"read_strategy":      d.readStrategy,
		"shards":             shardStatus,
		"total_nodes":        d.countTotalNodes(),
	}
}

// countTotalNodes counts total nodes across all shards
func (d *DistributedVectorDB) countTotalNodes() int {
	count := 0
	for _, nodes := range d.shards {
		count += len(nodes)
	}
	return count
}

// healthCheckLoop performs periodic health checks on all shard nodes
func (d *DistributedVectorDB) healthCheckLoop() {
	ticker := time.NewTicker(d.healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			d.performHealthChecks()
		case <-d.stopHealthCheck:
			return
		}
	}
}

// performHealthChecks checks health of all nodes
func (d *DistributedVectorDB) performHealthChecks() {
	d.mu.Lock()
	defer d.mu.Unlock()

	for _, nodes := range d.shards {
		for _, node := range nodes {
			// Perform health check
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			req, _ := http.NewRequestWithContext(ctx, "GET", node.HTTPAddr+"/health", nil)
			resp, err := d.httpClient.Do(req)
			cancel()

			if err != nil || resp.StatusCode != http.StatusOK {
				node.Healthy = false
			} else {
				node.Healthy = true
				node.LastSeen = time.Now()
			}

			if resp != nil {
				resp.Body.Close()
			}
		}
	}
}

// Shutdown stops the health check loop
func (d *DistributedVectorDB) Shutdown() {
	close(d.stopHealthCheck)
}
