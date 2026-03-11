package cluster

import (
	"fmt"
	"hash/fnv"
	"math"
	"testing"
)

func TestShardingStrategyDefaults(t *testing.T) {
	cfg := DistributedConfig{
		NumShards:         3,
		ReplicationFactor: 1,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	if d.shardingStrategy != ShardByCollection {
		t.Fatalf("expected default sharding strategy %q, got %q", ShardByCollection, d.shardingStrategy)
	}
}

func TestShardingStrategyVector(t *testing.T) {
	cfg := DistributedConfig{
		NumShards:         3,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByVector,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	if d.shardingStrategy != ShardByVector {
		t.Fatalf("expected sharding strategy %q, got %q", ShardByVector, d.shardingStrategy)
	}
}

func TestGetShardForVector(t *testing.T) {
	numShards := 5
	cfg := DistributedConfig{
		NumShards:         numShards,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByVector,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	// Deterministic: same ID always maps to same shard
	for _, id := range []string{"vec-1", "vec-2", "vec-3", "test-vector"} {
		shard1 := d.getShardForVector(id)
		shard2 := d.getShardForVector(id)
		if shard1 != shard2 {
			t.Fatalf("vector %q: non-deterministic sharding: %d vs %d", id, shard1, shard2)
		}
		if shard1 < 0 || shard1 >= numShards {
			t.Fatalf("vector %q: shard %d out of range [0, %d)", id, shard1, numShards)
		}
	}

	// Verify it uses FNV-64a
	h := fnv.New64a()
	h.Write([]byte("vec-1"))
	expected := int(h.Sum64() % uint64(numShards))
	got := d.getShardForVector("vec-1")
	if got != expected {
		t.Fatalf("expected shard %d for vec-1, got %d", expected, got)
	}
}

func TestGetShardsForInsertCollection(t *testing.T) {
	cfg := DistributedConfig{
		NumShards:         4,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByCollection,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	shards := d.getShardsForWrite("my-collection", "vec-123")
	if len(shards) != 1 {
		t.Fatalf("expected 1 shard, got %d", len(shards))
	}

	// Should match collection-based routing
	expected := d.getShardForCollection("my-collection")
	if shards[0] != expected {
		t.Fatalf("expected shard %d, got %d", expected, shards[0])
	}
}

func TestGetShardsForInsertVector(t *testing.T) {
	cfg := DistributedConfig{
		NumShards:         4,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByVector,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	shards := d.getShardsForWrite("my-collection", "vec-123")
	if len(shards) != 1 {
		t.Fatalf("expected 1 shard, got %d", len(shards))
	}

	// Should match vector-based routing (ignores collection)
	expected := d.getShardForVector("vec-123")
	if shards[0] != expected {
		t.Fatalf("expected shard %d, got %d", expected, shards[0])
	}
}

func TestGetShardsForQueryVector(t *testing.T) {
	numShards := 3
	cfg := DistributedConfig{
		NumShards:         numShards,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByVector,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	// Register shards so they show up
	for i := 0; i < numShards; i++ {
		d.RegisterShard(&ShardNode{NodeID: fmt.Sprintf("node-%d", i), ShardID: i, Role: RolePrimary, HTTPAddr: fmt.Sprintf("http://localhost:%d", 8000+i), Healthy: true})
	}

	// Vector sharding: queries scatter to ALL shards regardless of collection
	shards := d.getShardsForQuery([]string{"some-collection"})
	if len(shards) != numShards {
		t.Fatalf("vector sharding query should scatter to all %d shards, got %d", numShards, len(shards))
	}

	// Even with no collections specified
	shards = d.getShardsForQuery(nil)
	if len(shards) != numShards {
		t.Fatalf("vector sharding query with nil collections should scatter to all %d shards, got %d", numShards, len(shards))
	}
}

func TestGetShardsForQueryCollection(t *testing.T) {
	numShards := 4
	cfg := DistributedConfig{
		NumShards:         numShards,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByCollection,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	// Register shards
	for i := 0; i < numShards; i++ {
		d.RegisterShard(&ShardNode{NodeID: fmt.Sprintf("node-%d", i), ShardID: i, Role: RolePrimary, HTTPAddr: fmt.Sprintf("http://localhost:%d", 8000+i), Healthy: true})
	}

	// Collection sharding: queries go to specific shards
	shards := d.getShardsForQuery([]string{"collection-a"})
	if len(shards) != 1 {
		t.Fatalf("collection sharding with 1 collection should query 1 shard, got %d", len(shards))
	}

	// No collections: broadcast
	shards = d.getShardsForQuery(nil)
	if len(shards) != numShards {
		t.Fatalf("collection sharding with nil collections should broadcast to all %d shards, got %d", numShards, len(shards))
	}
}

func TestVectorShardingDistribution(t *testing.T) {
	numShards := 5
	numVectors := 10000
	cfg := DistributedConfig{
		NumShards:         numShards,
		ReplicationFactor: 1,
		ShardingStrategy:  ShardByVector,
		CoordinatorID:     "test",
	}
	d := NewDistributedVectorDB(cfg)
	defer d.Shutdown()

	// Count distribution
	counts := make(map[int]int, numShards)
	for i := 0; i < numVectors; i++ {
		id := fmt.Sprintf("vec-%d", i)
		shard := d.getShardForVector(id)
		counts[shard]++
	}

	// Verify all shards receive vectors
	for i := 0; i < numShards; i++ {
		if counts[i] == 0 {
			t.Fatalf("shard %d received 0 vectors", i)
		}
	}

	// Verify roughly uniform distribution (±30% of expected)
	expected := float64(numVectors) / float64(numShards)
	tolerance := expected * 0.30
	for shard, count := range counts {
		deviation := math.Abs(float64(count) - expected)
		if deviation > tolerance {
			t.Errorf("shard %d: got %d vectors, expected ~%.0f (±%.0f)", shard, count, expected, tolerance)
		}
	}

	t.Logf("Distribution across %d shards: %v", numShards, counts)
}
