package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

// ===========================================================================================
// DISTRIBUTED VECTORDB DEMO
// Demonstrates sharding, replication, and query aggregation
// ===========================================================================================

// DistributedDemo runs a demonstration of the distributed vectordb
func DistributedDemo() {
	fmt.Println("===== Distributed VectorDB Demo =====")

	// Wait for all services to be ready
	fmt.Println("Waiting for services to start...")
	time.Sleep(3 * time.Second)

	coordinatorAddr := "http://localhost:8080"

	// Check cluster status
	fmt.Println("\n1. Checking cluster status...")
	checkClusterStatus(coordinatorAddr)

	// Insert documents into different collections (will be sharded)
	fmt.Println("\n2. Inserting documents into multiple collections...")
	insertDemoData(coordinatorAddr)

	// Query within a single collection (single shard)
	fmt.Println("\n3. Querying single collection (single-shard query)...")
	querySingleCollection(coordinatorAddr)

	// Query across multiple collections (multi-shard query)
	fmt.Println("\n4. Querying multiple collections (multi-shard aggregation)...")
	queryMultipleCollections(coordinatorAddr)

	// Broadcast query (all shards)
	fmt.Println("\n5. Broadcast query across all shards...")
	broadcastQuery(coordinatorAddr)

	// Check metrics endpoint
	fmt.Println("\n6. Checking Prometheus metrics...")
	checkMetrics(coordinatorAddr)

	// Check failover stats
	fmt.Println("\n7. Checking failover status...")
	checkFailoverStats(coordinatorAddr)

	fmt.Println("\n===== Demo Complete =====")
	fmt.Println("\nProduction Features Demonstrated:")
	fmt.Println("  ✅ Collection-based sharding")
	fmt.Println("  ✅ Multi-shard query aggregation")
	fmt.Println("  ✅ Prometheus metrics at /metrics")
	fmt.Println("  ✅ Automatic failover monitoring")
	fmt.Println("  ✅ Health checking")
}

func checkClusterStatus(addr string) {
	resp, err := http.Get(addr + "/admin/cluster_status")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	var status map[string]any
	json.Unmarshal(body, &status)

	fmt.Printf("Cluster Status:\n")
	fmt.Printf("  Shards: %v\n", status["num_shards"])
	fmt.Printf("  Replication Factor: %v\n", status["replication_factor"])
	fmt.Printf("  Read Strategy: %v\n", status["read_strategy"])
	fmt.Printf("  Total Nodes: %v\n", status["total_nodes"])

	if shards, ok := status["shards"].([]any); ok {
		for _, s := range shards {
			shard := s.(map[string]any)
			fmt.Printf("\n  Shard %v:\n", shard["shard_id"])
			if nodes, ok := shard["nodes"].([]any); ok {
				for _, n := range nodes {
					node := n.(map[string]any)
					fmt.Printf("    - %s (%s): healthy=%v\n",
						node["node_id"], node["role"], node["healthy"])
				}
			}
		}
	}
}

func insertDemoData(addr string) {
	// Sample documents for different collections
	docs := []struct {
		Doc        string
		Collection string
		Meta       map[string]string
	}{
		// Customer A docs (will go to shard based on hash("customer-a"))
		{
			Doc:        "How to reset your password in the dashboard",
			Collection: "customer-a",
			Meta:       map[string]string{"category": "auth", "priority": "high"},
		},
		{
			Doc:        "Integrating the API with your application",
			Collection: "customer-a",
			Meta:       map[string]string{"category": "api", "priority": "medium"},
		},

		// Customer B docs (will go to different shard based on hash("customer-b"))
		{
			Doc:        "Troubleshooting connection errors",
			Collection: "customer-b",
			Meta:       map[string]string{"category": "networking", "priority": "high"},
		},
		{
			Doc:        "Understanding billing and pricing",
			Collection: "customer-b",
			Meta:       map[string]string{"category": "billing", "priority": "low"},
		},

		// Customer C docs (will go to different shard based on hash("customer-c"))
		{
			Doc:        "Security best practices for API keys",
			Collection: "customer-c",
			Meta:       map[string]string{"category": "security", "priority": "critical"},
		},
		{
			Doc:        "Monitoring and analytics dashboard guide",
			Collection: "customer-c",
			Meta:       map[string]string{"category": "monitoring", "priority": "medium"},
		},
	}

	client := &http.Client{Timeout: 10 * time.Second}

	for i, doc := range docs {
		reqBody, _ := json.Marshal(map[string]any{
			"doc":        doc.Doc,
			"collection": doc.Collection,
			"meta":       doc.Meta,
		})

		resp, err := client.Post(addr+"/insert", "application/json", bytes.NewReader(reqBody))
		if err != nil {
			fmt.Printf("Error inserting doc %d: %v\n", i, err)
			continue
		}

		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()

		var result map[string]any
		json.Unmarshal(body, &result)

		fmt.Printf("  Inserted: %s -> collection '%s' (id: %s)\n",
			doc.Doc[:40]+"...", doc.Collection, result["id"])
	}
}

func querySingleCollection(addr string) {
	query := map[string]any{
		"query":       "password reset authentication",
		"top_k":       3,
		"collections": []string{"customer-a"},
		"mode":        "hybrid",
	}

	results := performQuery(addr, query)

	fmt.Printf("Query: '%s' in collection 'customer-a'\n", query["query"])
	fmt.Printf("Results: %d documents\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. [%.2f] %s\n", i+1, result["score"], result["doc"].(string)[:60])
	}
}

func queryMultipleCollections(addr string) {
	query := map[string]any{
		"query":       "troubleshooting errors",
		"top_k":       5,
		"collections": []string{"customer-a", "customer-b", "customer-c"},
		"mode":        "hybrid",
	}

	results := performQuery(addr, query)

	fmt.Printf("Query: '%s' across 3 collections\n", query["query"])
	fmt.Printf("Results: %d documents (aggregated from multiple shards)\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. [%.2f] %s\n", i+1, result["score"], result["doc"].(string)[:60])
	}
}

func broadcastQuery(addr string) {
	query := map[string]any{
		"query":       "api security best practices",
		"top_k":       10,
		"collections": []string{}, // Empty = broadcast to all shards
		"mode":        "hybrid",
	}

	results := performQuery(addr, query)

	fmt.Printf("Query: '%s' (broadcast to all shards)\n", query["query"])
	fmt.Printf("Results: %d documents (aggregated from ALL shards)\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. [%.2f] %s\n", i+1, result["score"], result["doc"].(string)[:60])
	}
}

func performQuery(addr string, query map[string]any) []map[string]any {
	reqBody, _ := json.Marshal(query)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Post(addr+"/query", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		fmt.Printf("Error querying: %v\n", err)
		return nil
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)

	var result struct {
		IDs    []string            `json:"ids"`
		Docs   []string            `json:"docs"`
		Scores []float64           `json:"scores"`
		Meta   []map[string]string `json:"meta"`
	}
	json.Unmarshal(body, &result)

	// Convert to array of maps for easier handling
	results := make([]map[string]any, len(result.IDs))
	for i := range result.IDs {
		results[i] = map[string]any{
			"id":    result.IDs[i],
			"doc":   result.Docs[i],
			"score": result.Scores[i],
		}
		if i < len(result.Meta) {
			results[i]["meta"] = result.Meta[i]
		}
	}

	return results
}

func checkMetrics(addr string) {
	resp, err := http.Get(addr + "/metrics")
	if err != nil {
		fmt.Printf("Error: %v (metrics may not be enabled)\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Metrics endpoint not available (status %d)\n", resp.StatusCode)
		return
	}

	body, _ := ioutil.ReadAll(resp.Body)
	lines := bytes.Split(body, []byte("\n"))

	fmt.Println("Prometheus Metrics Available:")

	// Show sample metrics (first 10 lines that aren't comments)
	count := 0
	for _, line := range lines {
		if len(line) > 0 && line[0] != '#' {
			fmt.Printf("  %s\n", string(line))
			count++
			if count >= 5 {
				break
			}
		}
	}

	fmt.Printf("  ... (%d more metrics available at /metrics)\n", len(lines)-count-20)
}

func checkFailoverStats(addr string) {
	resp, err := http.Get(addr + "/admin/failover_stats")
	if err != nil {
		fmt.Printf("Error: %v (failover may not be enabled)\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		fmt.Printf("Failover not available: %s\n", string(body))
		return
	}

	body, _ := ioutil.ReadAll(resp.Body)
	var stats map[string]any
	json.Unmarshal(body, &stats)

	fmt.Printf("Failover Manager Status:\n")
	fmt.Printf("  Enabled: %v\n", stats["enabled"])
	fmt.Printf("  Unhealthy Threshold: %v\n", stats["unhealthy_threshold"])
	fmt.Printf("  Check Interval: %v\n", stats["check_interval"])

	if shards, ok := stats["shards"].([]any); ok && len(shards) > 0 {
		fmt.Printf("  Monitored Shards: %d\n", len(shards))
		for _, s := range shards {
			shard := s.(map[string]any)
			fmt.Printf("    - Shard %v: failover_in_progress=%v\n",
				shard["shard_id"], shard["failover_in_progress"])
		}
	} else {
		fmt.Println("  No shards being monitored yet")
	}
}
