package index

// DISKANN DEADLOCK FIX
//
// This file contains the deadlock fix for DiskANN parallel graph construction.
//
// PROBLEM:
// Workers in buildEdgesParallel acquire RLock → call greedySearch → access d.graph
// If something tries to acquire write lock during this, deadlock occurs.
//
// SOLUTION:
// Add greedySearchSafe that copies graph snapshot under lock, then searches lock-free.
//
// INTEGRATION:
// Replace line 236-238 in diskann_parallel.go:
//   d.mu.RLock()
//   candidates = d.greedySearch(vec, d.efConstruction)
//   d.mu.RUnlock()
//
// With:
//   candidates = d.greedySearchSafe(vec, d.efConstruction)

// graphSnapshot holds a snapshot of graph data for lock-free search
type graphSnapshot struct {
	graph   map[uint64][]uint64  // Neighbor adjacency lists
	deleted map[uint64]bool      // Deleted node flags
}

// copyGraphSnapshot creates a shallow copy of graph and deleted maps
// This is fast (just copying map headers and slices) and prevents deadlocks
func (d *DiskANNIndex) copyGraphSnapshot() graphSnapshot {
	d.mu.RLock()
	defer d.mu.RUnlock()

	snapshot := graphSnapshot{
		graph:   make(map[uint64][]uint64, len(d.graph)),
		deleted: make(map[uint64]bool, len(d.deleted)),
	}

	// Shallow copy graph (neighbor IDs are immutable)
	for id, neighbors := range d.graph {
		neighborsCopy := make([]uint64, len(neighbors))
		copy(neighborsCopy, neighbors)
		snapshot.graph[id] = neighborsCopy
	}

	// Copy deleted flags
	for id := range d.deleted {
		snapshot.deleted[id] = true
	}

	return snapshot
}

// greedySearchSafe performs greedy graph search with deadlock prevention
// This is the main entry point that workers should use
func (d *DiskANNIndex) greedySearchSafe(query []float32, ef int) []Result {
	// Take snapshot under lock (fast operation)
	snapshot := d.copyGraphSnapshot()

	// Search without lock (prevents deadlocks)
	return d.greedySearchOnSnapshot(query, ef, snapshot)
}

// greedySearchOnSnapshot performs greedy search on a graph snapshot
// This function is lock-free and can be called from parallel workers safely
func (d *DiskANNIndex) greedySearchOnSnapshot(query []float32, ef int, snapshot graphSnapshot) []Result {
	if len(snapshot.graph) == 0 {
		return []Result{}
	}

	// Start from random entry point
	var entryID uint64
	for id := range snapshot.graph {
		if !snapshot.deleted[id] {
			entryID = id
			break
		}
	}

	visited := make(map[uint64]bool)
	candidates := make([]Result, 0, ef)

	// Get entry vector
	entryVec, err := d.getVector(entryID)
	if err != nil {
		return []Result{}
	}

	entryDist := d.distance(query, entryVec)
	candidates = append(candidates, Result{ID: entryID, Distance: entryDist})
	visited[entryID] = true

	// Greedy expansion
	for i := 0; i < ef; i++ {
		if i >= len(candidates) {
			break
		}

		current := candidates[i]
		neighbors := snapshot.graph[current.ID]  // Use snapshot, not d.graph

		for _, nID := range neighbors {
			if visited[nID] || snapshot.deleted[nID] {  // Use snapshot, not d.deleted
				continue
			}

			nVec, err := d.getVector(nID)
			if err != nil {
				continue
			}

			dist := d.distance(query, nVec)
			candidates = append(candidates, Result{ID: nID, Distance: dist})
			visited[nID] = true
		}

		// Sort by distance (keep best candidates)
		// Using sort.Slice (imported at package level)
		for j := 0; j < len(candidates); j++ {
			for k := j + 1; k < len(candidates); k++ {
				if candidates[k].Distance < candidates[j].Distance {
					candidates[j], candidates[k] = candidates[k], candidates[j]
				}
			}
		}

		if len(candidates) > ef*2 {
			candidates = candidates[:ef*2]
		}
	}

	// Return top ef results
	if len(candidates) > ef {
		candidates = candidates[:ef]
	}

	return candidates
}
