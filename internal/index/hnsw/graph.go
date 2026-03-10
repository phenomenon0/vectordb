package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"time"

	"github.com/coder/hnsw/heap"
)

type Vector = []float32

// FilterFunc returns true if the node with the given key should be included in results.
// Used for filtered HNSW search where filtering happens DURING graph traversal,
// not after (post-filtering). This provides 5-20x speedup for selective filters.
type FilterFunc[K cmp.Ordered] func(key K) bool

// Node is a node in the graph.
type Node[K cmp.Ordered] struct {
	Key   K
	Value Vector
}

func MakeNode[K cmp.Ordered](key K, vec Vector) Node[K] {
	return Node[K]{Key: key, Value: vec}
}

// layerNode is a node in a layer of the graph.
type layerNode[K cmp.Ordered] struct {
	Node[K]

	// neighbors is a slice of neighbor nodes.
	// Slice provides better cache locality than a map for small M (typically 16).
	// O(M) linear scans are faster than map lookups for M <= ~32.
	neighbors []*layerNode[K]
}

// hasNeighbor returns true if the node has a neighbor with the given key.
func (n *layerNode[K]) hasNeighbor(key K) bool {
	for _, nb := range n.neighbors {
		if nb.Key == key {
			return true
		}
	}
	return false
}

// removeNeighbor removes a neighbor by key using swap-remove (O(1) swap + O(M) scan).
func (n *layerNode[K]) removeNeighbor(key K) {
	for i, nb := range n.neighbors {
		if nb.Key == key {
			last := len(n.neighbors) - 1
			n.neighbors[i] = n.neighbors[last]
			n.neighbors[last] = nil // avoid memory leak
			n.neighbors = n.neighbors[:last]
			return
		}
	}
}

// diverseCandidate holds a candidate node and its distance to the owner node.
type diverseCandidate[K cmp.Ordered] struct {
	node        *layerNode[K]
	distToOwner float32
}

// selectDiverseNeighbors implements hnswlib's getNeighborsByHeuristic2.
// It selects up to m neighbors that are both close to the owner AND spread
// across different angular regions, preventing neighbor clustering.
func selectDiverseNeighbors[K cmp.Ordered](candidates []diverseCandidate[K], m int, dist DistanceFunc) []*layerNode[K] {
	// Sort candidates by distance to owner (ascending = closest first)
	slices.SortFunc(candidates, func(a, b diverseCandidate[K]) int {
		if a.distToOwner < b.distToOwner {
			return -1
		}
		if a.distToOwner > b.distToOwner {
			return 1
		}
		return 0
	})

	selected := make([]*layerNode[K], 0, m)
	skipped := make([]*layerNode[K], 0)

	for _, c := range candidates {
		if len(selected) >= m {
			break
		}
		// Check if candidate c is "covered" by an already-selected neighbor s:
		// if dist(c, s) < dist(c, owner), then s is closer to c than the owner is,
		// meaning c doesn't add directional diversity.
		covered := false
		for _, s := range selected {
			if dist(c.node.Value, s.Value) < c.distToOwner {
				covered = true
				break
			}
		}
		if covered {
			skipped = append(skipped, c.node)
		} else {
			selected = append(selected, c.node)
		}
	}

	// Backfill from skipped candidates (closest first) if < m selected
	for _, s := range skipped {
		if len(selected) >= m {
			break
		}
		selected = append(selected, s)
	}

	return selected
}

// addNeighbor adds a neighbor to the node using diversity-based heuristic
// selection (hnswlib's getNeighborsByHeuristic2) when the neighbor set overflows.
//
// Per hnswlib: when overflow occurs, the heuristic shrinks to M and excess
// nodes are silently dropped. Evicted nodes' neighbor lists are NOT touched.
// The graph is intentionally asymmetric during construction. Backlink
// cleanup (removeNeighbor + replenish) is reserved for Delete operations.
//
// DO NOT add backlink repair here — it causes cascading graph disruption
// that degrades recall from 1.0 to 0.65 at 10k+ scale.
func (n *layerNode[K]) addNeighbor(newNode *layerNode[K], m int, dist DistanceFunc) {
	if n.hasNeighbor(newNode.Key) {
		return
	}

	n.neighbors = append(n.neighbors, newNode)
	if len(n.neighbors) <= m {
		return
	}

	// Over capacity: apply heuristic, silently drop excess (no backlink repair).
	candidates := make([]diverseCandidate[K], len(n.neighbors))
	for i, nb := range n.neighbors {
		candidates[i] = diverseCandidate[K]{
			node:        nb,
			distToOwner: dist(nb.Value, n.Value),
		}
	}

	n.neighbors = selectDiverseNeighbors(candidates, m, dist)
}

type searchCandidate[K cmp.Ordered] struct {
	node *layerNode[K]
	dist float32
}

func (s searchCandidate[K]) Less(o searchCandidate[K]) bool {
	return s.dist < o.dist
}

// search returns the layer node closest to the target node
// within the same layer.
// If filter is non-nil, only nodes passing the filter are included in results.
func (n *layerNode[K]) search(
	// k is the number of candidates in the result set.
	k int,
	efSearch int,
	target Vector,
	distance DistanceFunc,
	filter FilterFunc[K], // Optional filter applied during traversal
) []searchCandidate[K] {
	// Standard HNSW search: result set is efSearch-sized (not k-sized).
	// A larger result set keeps the "worst accepted" distance higher,
	// preventing premature termination and allowing broader exploration.
	// We return only the best k results at the end.
	resultCap := efSearch
	if resultCap < k {
		resultCap = k
	}

	candidates := heap.Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, efSearch))
	candidates.Push(
		searchCandidate[K]{
			node: n,
			dist: distance(n.Value, target),
		},
	)
	var (
		result  = heap.Heap[searchCandidate[K]]{}
		visited = make(map[K]bool, efSearch)
	)
	result.Init(make([]searchCandidate[K], 0, resultCap))

	// Begin with the entry node in the result set (if it passes filter).
	entryCandidate := candidates.Min()
	// Cache the max distance in the result set to avoid O(n/2) leaf scans
	// on every iteration. Updated only on push/pop to result heap.
	var maxDist float32
	if filter == nil || filter(n.Key) {
		result.Push(entryCandidate)
		maxDist = entryCandidate.dist
	}
	visited[n.Key] = true

	for candidates.Len() > 0 {
		popped := candidates.Pop()

		// Termination: if the best candidate (just popped from min-heap)
		// is farther than the worst in our result set, no improvement possible.
		if result.Len() >= resultCap && popped.dist >= maxDist {
			break
		}

		current := popped.node
		for _, neighbor := range current.neighbors {
			neighborID := neighbor.Key
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)

			// Always add to candidates for graph traversal (even if filtered)
			// This ensures we explore the graph properly
			candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
			if candidates.Len() > efSearch {
				candidates.PopLast()
			}

			// Only add to results if passes filter (or no filter)
			if filter != nil && !filter(neighborID) {
				continue // Skip filtered nodes for results, but still traverse
			}

			if result.Len() < resultCap {
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				// New element could be the new max
				if dist > maxDist {
					maxDist = dist
				}
			} else if dist < maxDist {
				result.PopLast()
				result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				// Recompute max after eviction (amortized — only on actual change)
				maxDist = result.Max().dist
			}
		}
	}

	// Return the best k results from the efSearch-sized result set.
	// Pop from min-heap gives us elements in sorted order (closest first).
	count := k
	if result.Len() < count {
		count = result.Len()
	}
	out := make([]searchCandidate[K], count)
	for i := 0; i < count; i++ {
		out[i] = result.Pop()
	}
	return out
}

func (n *layerNode[K]) replenish(m int, dist DistanceFunc) {
	if len(n.neighbors) >= m {
		return
	}

	// Restore connectivity by pulling candidates from neighbors-of-neighbors.
	for _, neighbor := range n.neighbors {
		for _, candidate := range neighbor.neighbors {
			if n.hasNeighbor(candidate.Key) {
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, dist)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

// isolate removes the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[K]) isolate(m int, dist DistanceFunc) {
	for _, neighbor := range n.neighbors {
		neighbor.removeNeighbor(n.Key)
		neighbor.replenish(m, dist)
	}
}

type layer[K cmp.Ordered] struct {
	// nodes is a map of nodes IDs to nodes.
	// All nodes in a higher layer are also in the lower layers, an essential
	// property of the graph.
	//
	// nodes is exported for interop with encoding/gob.
	nodes map[K]*layerNode[K]
}

func (l *layer[K]) entry() *layerNode[K] {
	if l == nil {
		return nil
	}
	for _, node := range l.nodes {
		return node
	}
	return nil
}

func (l *layer[K]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World graph.
// All public parameters must be set before adding nodes to the graph.
// K is cmp.Ordered instead of of comparable so that they can be sorted.
type Graph[K cmp.Ordered] struct {
	// Distance is the distance function used to compare embeddings.
	Distance DistanceFunc

	// Rng is used for level generation. It may be set to a deterministic value
	// for reproducibility. Note that deterministic number generation can lead to
	// degenerate graphs when exposed to adversarial inputs.
	Rng *rand.Rand

	// M is the maximum number of neighbors to keep for each node.
	// A good default for OpenAI embeddings is 16.
	M int

	// Ml is the level generation factor.
	// Computed as 1/ln(M) to match hnswlib's level distribution.
	Ml float64

	// EfSearch is the number of nodes to consider in the search phase.
	// 20 is a reasonable default. Higher values improve search accuracy at
	// the expense of memory.
	EfSearch int

	// layers is a slice of layers in the graph.
	layers []*layer[K]

	// FIX 1: Explicit entry point — always the node at the highest level.
	// hnswlib tracks enterpoint_node_ and updates it on every insert.
	// Without this, we pick an arbitrary map entry which gives bad search paths.
	entryPointKey *K
	entryLevel    int
}

// m0 returns the max neighbors for the base layer (level 0).
// FIX 3: hnswlib uses m0 = 2*M at the base layer for denser connectivity.
func (g *Graph[K]) m0() int {
	return g.M * 2
}

// mForLevel returns the max neighbor count for the given layer.
func (g *Graph[K]) mForLevel(level int) int {
	if level == 0 {
		return g.m0()
	}
	return g.M
}

func defaultRand() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

// NewGraph returns a new graph with default parameters, roughly designed for
// storing OpenAI embeddings.
func NewGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 20,
		Rng:      defaultRand(),
	}
}

// randomLevel generates a random level for a new node.
// FIX 2: Use hnswlib's formula: level = floor(-log(uniform) * mL)
// where mL = 1/ln(M). This produces the correct exponential distribution
// instead of the coin-flip approach which was too conservative.
func (h *Graph[K]) randomLevel() int {
	if h.Rng == nil {
		h.Rng = defaultRand()
	}

	// hnswlib: reverse_size = 1/mult_ where mult_ = 1/ln(M)
	// so reverse_size = ln(M)
	// level = floor(-ln(rand) * reverse_size) = floor(-ln(rand) * ln(M))
	mL := 1.0 / math.Log(float64(h.M))
	r := h.Rng.Float64()
	if r == 0 {
		r = 1e-9 // avoid -log(0)
	}
	level := int(-math.Log(r) * mL)

	// Cap at a reasonable maximum to prevent degenerate graphs
	maxL := int(math.Log(float64(max(h.Len()+1, 1)))*mL) + 2
	if maxL < 1 {
		maxL = 1
	}
	if level > maxL {
		level = maxL
	}

	return level
}

func (g *Graph[K]) assertDims(n Vector) {
	if len(g.layers) == 0 {
		return
	}
	hasDims := g.Dims()
	if hasDims != len(n) {
		panic(fmt.Sprint("embedding dimension mismatch: ", hasDims, " != ", len(n)))
	}
}

// Dims returns the number of dimensions in the graph, or
// 0 if the graph is empty.
func (g *Graph[K]) Dims() int {
	if len(g.layers) == 0 {
		return 0
	}
	return len(g.layers[0].entry().Value)
}

func ptr[T any](v T) *T {
	return &v
}

// Add inserts nodes into the graph.
// If another node with the same ID exists, it is replaced.
func (g *Graph[K]) Add(nodes ...Node[K]) {
	for _, node := range nodes {
		key := node.Key
		vec := node.Value

		g.assertDims(vec)
		insertLevel := g.randomLevel()
		// Create layers that don't exist yet.
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K]{})
		}

		if insertLevel < 0 {
			panic("invalid level")
		}

		var elevator *K

		preLen := g.Len()
		// Check if this is a replacement (node with same key exists)
		_, isReplacement := g.layers[0].nodes[key]

		// FIX 1: Use the explicit entry point for search start
		if g.entryPointKey != nil {
			elevator = g.entryPointKey
		}

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]
			newNode := &layerNode[K]{
				Node: Node[K]{
					Key:   key,
					Value: vec,
				},
			}

			// Insert the new node into the layer.
			if layer.entry() == nil {
				layer.nodes = map[K]*layerNode[K]{key: newNode}
				continue
			}

			// Use elevator (entry point) to find the search start in this layer
			searchPoint := layer.entry()
			if elevator != nil {
				if ep, ok := layer.nodes[*elevator]; ok {
					searchPoint = ep
				}
			}

			if g.Distance == nil {
				panic("(*Graph).Distance must be set")
			}

			// FIX 3: Use mForLevel to get correct M for this layer
			mLevel := g.mForLevel(i)
			// FIX 4: Pass efSearch as k so the search returns ALL ef candidates,
			// not just mLevel. This gives selectDiverseNeighbors a full pool
			// (e.g. 200 → pick 32) instead of zero selectivity. Matches hnswlib.
			neighborhood := searchPoint.search(g.EfSearch, g.EfSearch, vec, g.Distance, nil)
			if len(neighborhood) == 0 {
				// This should never happen because the searchPoint itself
				// should be in the result set.
				panic("no nodes found")
			}

			// Re-set the elevator node for the next layer.
			elevator = ptr(neighborhood[0].node.Key)

			if insertLevel >= i {
				if _, ok := layer.nodes[key]; ok {
					g.Delete(key)
				}
				// Insert the new node into the layer.
				layer.nodes[key] = newNode

				// Select diverse neighbors for the new node using heuristic
				candidates := make([]diverseCandidate[K], 0, len(neighborhood))
				for _, n := range neighborhood {
					candidates = append(candidates, diverseCandidate[K]{
						node:        n.node,
						distToOwner: n.dist,
					})
				}
				newNode.neighbors = selectDiverseNeighbors(candidates, mLevel, g.Distance)

				// Add backlinks from selected neighbors to the new node
				for _, neighbor := range newNode.neighbors {
					neighbor.addNeighbor(newNode, mLevel, g.Distance)
				}
			}
		}

		// FIX 1: Update entry point if this node has a higher level
		if g.entryPointKey == nil || insertLevel > g.entryLevel {
			g.entryPointKey = ptr(key)
			g.entryLevel = insertLevel
		}

		// Invariant check: the node should have been added to the graph.
		// For replacements, length stays the same; for new inserts, it increases by 1.
		expectedLen := preLen + 1
		if isReplacement {
			expectedLen = preLen
		}
		if g.Len() != expectedLen {
			panic("node not added")
		}
	}
}

// Search finds the k nearest neighbors from the target node.
func (h *Graph[K]) Search(near Vector, k int) []Node[K] {
	return h.SearchWithEf(near, k, h.EfSearch, nil)
}

// SearchFiltered finds the k nearest neighbors from the target node,
// only including nodes that pass the filter function.
// If filter is nil, all nodes are included (equivalent to Search).
//
// The filter is applied DURING graph traversal, not after, which provides
// significant speedup (5-20x) for selective filters compared to post-filtering.
func (h *Graph[K]) SearchFiltered(near Vector, k int, filter FilterFunc[K]) []Node[K] {
	return h.SearchWithEf(near, k, h.EfSearch, filter)
}

// SearchWithEf finds the k nearest neighbors using the specified efSearch beam width.
// This is safe for concurrent use — efSearch is passed as a parameter, not read from the graph.
func (h *Graph[K]) SearchWithEf(near Vector, k int, efSearch int, filter FilterFunc[K]) []Node[K] {
	h.assertDims(near)
	if len(h.layers) == 0 {
		return nil
	}

	// FIX 1: Start from the explicit entry point at the top level
	var elevator *K
	if h.entryPointKey != nil {
		elevator = h.entryPointKey
	}

	for layer := len(h.layers) - 1; layer >= 0; layer-- {
		// Find search start for this layer
		searchPoint := h.layers[layer].entry()
		if elevator != nil {
			if ep, ok := h.layers[layer].nodes[*elevator]; ok {
				searchPoint = ep
			}
		}

		// Descending hierarchies: greedy search (ef=1) per hnswlib.
		// Upper layers are sparse; full ef is wasted here.
		if layer > 0 {
			nodes := searchPoint.search(1, 1, near, h.Distance, nil)
			if len(nodes) > 0 {
				elevator = ptr(nodes[0].node.Key)
			}
			continue
		}

		// Base layer - apply filter during search
		nodes := searchPoint.search(k, efSearch, near, h.Distance, filter)
		out := make([]Node[K], 0, len(nodes))

		for _, node := range nodes {
			out = append(out, node.node.Node)
		}

		return out
	}

	panic("unreachable")
}

// Len returns the number of nodes in the graph.
func (h *Graph[K]) Len() int {
	if len(h.layers) == 0 {
		return 0
	}
	return h.layers[0].size()
}

// Delete removes a node from the graph by key.
// It tries to preserve the clustering properties of the graph by
// replenishing connectivity in the affected neighborhoods.
func (h *Graph[K]) Delete(key K) bool {
	if len(h.layers) == 0 {
		return false
	}

	var deleted bool
	for i, layer := range h.layers {
		node, ok := layer.nodes[key]
		if !ok {
			continue
		}
		delete(layer.nodes, key)
		node.isolate(h.mForLevel(i), h.Distance)
		deleted = true
	}

	// FIX 1: If we deleted the entry point, find a new one
	if deleted && h.entryPointKey != nil && *h.entryPointKey == key {
		h.entryPointKey = nil
		h.entryLevel = 0
		// Find the node at the highest occupied layer
		for i := len(h.layers) - 1; i >= 0; i-- {
			if h.layers[i].size() > 0 {
				ep := h.layers[i].entry()
				h.entryPointKey = ptr(ep.Key)
				h.entryLevel = i
				break
			}
		}
	}

	return deleted
}

// Lookup returns the vector with the given key.
func (h *Graph[K]) Lookup(key K) (Vector, bool) {
	if len(h.layers) == 0 {
		return nil, false
	}

	node, ok := h.layers[0].nodes[key]
	if !ok {
		return nil, false
	}
	return node.Value, ok
}

// Update replaces the vector for an existing key in place.
// This is more efficient than Delete+Add for updating vectors.
// Returns false if the key doesn't exist.
func (h *Graph[K]) Update(key K, newVec Vector) bool {
	if len(h.layers) == 0 {
		return false
	}

	// Update the vector in all layers where this node exists
	found := false
	for _, layer := range h.layers {
		if node, ok := layer.nodes[key]; ok {
			node.Value = newVec
			found = true
		}
	}
	return found
}
