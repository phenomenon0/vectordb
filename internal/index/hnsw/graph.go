package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

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

	// _neighbors stores a pointer to []*layerNode[K] (the neighbor slice header).
	// Accessed atomically for concurrent read safety during AddConcurrent.
	// Writers must hold the appropriate node lock from nodeLockPool.
	_neighbors unsafe.Pointer // *[]*layerNode[K]
}

// loadNeighbors atomically loads the neighbors slice.
// Safe for concurrent reads during AddConcurrent search traversal.
func (n *layerNode[K]) loadNeighbors() []*layerNode[K] {
	p := atomic.LoadPointer(&n._neighbors)
	if p == nil {
		return nil
	}
	return *(*[]*layerNode[K])(p)
}

// storeNeighbors atomically publishes a new neighbors slice.
// Must be called under the appropriate node lock for writes.
func (n *layerNode[K]) storeNeighbors(nb []*layerNode[K]) {
	atomic.StorePointer(&n._neighbors, unsafe.Pointer(&nb))
}

// hasNeighbor returns true if the node has a neighbor with the given key.
func (n *layerNode[K]) hasNeighbor(key K) bool {
	for _, nb := range n.loadNeighbors() {
		if nb.Key == key {
			return true
		}
	}
	return false
}

// removeNeighbor removes a neighbor by key using swap-remove (O(1) swap + O(M) scan).
func (n *layerNode[K]) removeNeighbor(key K) {
	nb := n.loadNeighbors()
	for i, node := range nb {
		if node.Key == key {
			last := len(nb) - 1
			nb[i] = nb[last]
			nb[last] = nil // avoid memory leak
			nb = nb[:last]
			n.storeNeighbors(nb)
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

	// Backfill from skipped candidates (closest first) if < m selected.
	// This ensures connectivity — nodes need enough neighbors for navigability.
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

	nb := append(n.loadNeighbors(), newNode)
	if len(nb) <= m {
		n.storeNeighbors(nb)
		return
	}

	// Over capacity: apply heuristic, silently drop excess (no backlink repair).
	candidates := make([]diverseCandidate[K], len(nb))
	for i, node := range nb {
		candidates[i] = diverseCandidate[K]{
			node:        node,
			distToOwner: dist(node.Value, n.Value),
		}
	}

	n.storeNeighbors(selectDiverseNeighbors(candidates, m, dist))
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

		// Per hnswlib: terminate when the closest candidate is strictly farther
		// than the worst result AND the result set is full.
		if popped.dist > maxDist && result.Len() >= resultCap {
			break
		}

		current := popped.node
		for _, neighbor := range current.loadNeighbors() {
			neighborID := neighbor.Key
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			dist := distance(neighbor.Value, target)

			// Per hnswlib: only add to candidate/result sets if the neighbor
			// is closer than the worst result OR the result set isn't full.
			// This keeps the candidate set focused on promising nodes.
			// For filtered search, always add to candidates for traversal.
			if filter != nil {
				// Filtered mode: always add to candidates for graph traversal
				candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})

				if !filter(neighborID) {
					continue
				}
				if result.Len() < resultCap {
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
					if dist > maxDist {
						maxDist = dist
					}
				} else if dist < maxDist {
					result.PopLast()
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
					maxDist = result.Max().dist
				}
			} else {
				// Unfiltered mode: match hnswlib exactly.
				// Only process if promising (closer than worst) or result not full.
				if dist < maxDist || result.Len() < resultCap {
					candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
					if result.Len() > resultCap {
						result.PopLast()
					}
					maxDist = result.Max().dist
				}
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

// greedyClosest performs a greedy walk to find the closest node to the target.
// This matches hnswlib's upper-layer traversal: at each node, scan all neighbors
// and move to the closest one. Repeat until no improvement.
func (n *layerNode[K]) greedyClosest(target Vector, distance DistanceFunc) *layerNode[K] {
	current := n
	currentDist := distance(n.Value, target)
	for {
		improved := false
		for _, neighbor := range current.loadNeighbors() {
			d := distance(neighbor.Value, target)
			if d < currentDist {
				current = neighbor
				currentDist = d
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	return current
}

func (n *layerNode[K]) replenish(m int, dist DistanceFunc) {
	if len(n.loadNeighbors()) >= m {
		return
	}

	// Restore connectivity by pulling candidates from neighbors-of-neighbors.
	for _, neighbor := range n.loadNeighbors() {
		for _, candidate := range neighbor.loadNeighbors() {
			if n.hasNeighbor(candidate.Key) {
				continue
			}
			if candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, dist)
			if len(n.loadNeighbors()) >= m {
				return
			}
		}
	}
}

// isolate removes the node from the graph by removing all connections
// to neighbors.
func (n *layerNode[K]) isolate(m int, dist DistanceFunc) {
	for _, neighbor := range n.loadNeighbors() {
		neighbor.removeNeighbor(n.Key)
		neighbor.replenish(m, dist)
	}
}

type layer[K cmp.Ordered] struct {
	mu sync.RWMutex
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

// arbitraryNode returns any node from the layer. Must be called with at least RLock held.
func (l *layer[K]) arbitraryNode() *layerNode[K] {
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

	// Concurrency support for AddConcurrent.
	entryMu   sync.RWMutex       // protects entryPointKey, entryLevel, layers slice growth
	nodeLocks nodeLockPool[K]    // per-node neighbor list locks (sharded by pointer)
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

	// No artificial level cap. The exponential distribution naturally produces
	// the correct level distribution — high levels are exponentially rare.
	// hnswlib does not cap levels.

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
// 0 if the graph is empty. Safe for concurrent use with AddConcurrent.
func (g *Graph[K]) Dims() int {
	g.entryMu.RLock()
	if len(g.layers) == 0 {
		g.entryMu.RUnlock()
		return 0
	}
	base := g.layers[0]
	g.entryMu.RUnlock()

	base.mu.RLock()
	e := base.arbitraryNode()
	base.mu.RUnlock()
	if e == nil {
		return 0
	}
	return len(e.Value)
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

		if g.Distance == nil {
			panic("(*Graph).Distance must be set")
		}

		// Insert node at each layer, beginning with the highest.
		for i := len(g.layers) - 1; i >= 0; i-- {
			layer := g.layers[i]

			// UPPER LAYERS (above insertLevel): greedy walk to find entry point.
			// Per hnswlib: upper layers do a simple greedy descent — at each node,
			// move to the closest neighbor until no improvement. No insertion.
			if i > insertLevel {
				if layer.entry() == nil {
					continue
				}
				searchPoint := layer.entry()
				if elevator != nil {
					if ep, ok := layer.nodes[*elevator]; ok {
						searchPoint = ep
					}
				}
				closest := searchPoint.greedyClosest(vec, g.Distance)
				elevator = ptr(closest.Key)
				continue
			}

			// INSERTION LAYERS (at or below insertLevel): full ef_construction search.
			newNode := &layerNode[K]{
				Node: Node[K]{
					Key:   key,
					Value: vec,
				},
			}

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

			mLevel := g.mForLevel(i)
			neighborhood := searchPoint.search(g.EfSearch, g.EfSearch, vec, g.Distance, nil)
			if len(neighborhood) == 0 {
				panic("no nodes found")
			}

			// Re-set the elevator node for the next layer.
			elevator = ptr(neighborhood[0].node.Key)

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
			newNode.storeNeighbors(selectDiverseNeighbors(candidates, mLevel, g.Distance))

			// Add backlinks from selected neighbors to the new node
			for _, neighbor := range newNode.loadNeighbors() {
				neighbor.addNeighbor(newNode, mLevel, g.Distance)
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

// randomLevelWith generates a random level using the provided Rng.
// This avoids contention on the shared Graph.Rng during concurrent insertion.
func (g *Graph[K]) randomLevelWith(rng *rand.Rand) int {
	mL := 1.0 / math.Log(float64(g.M))
	r := rng.Float64()
	if r == 0 {
		r = 1e-9
	}
	return int(-math.Log(r) * mL)
}

// AddConcurrent inserts a single node into the graph using fine-grained locking.
// Safe for concurrent calls from multiple goroutines. Each caller must provide
// its own *rand.Rand to avoid contention on the shared Rng.
//
// Locking protocol (matches hnswlib):
//  1. Layer growth and entry point: entryMu (write-locked only when growing layers or updating entry)
//  2. Layer node map: layer.mu (write for insertion, read for lookup)
//  3. Neighbor list mutation: nodeLocks per-node (only the node being mutated)
//  4. Neighbor list reads during search: lock-free (stale reads acceptable, same as hnswlib)
func (g *Graph[K]) AddConcurrent(node Node[K], rng *rand.Rand) {
	key := node.Key
	vec := node.Value

	insertLevel := g.randomLevelWith(rng)
	if insertLevel < 0 {
		panic("invalid level")
	}

	// Grow layers if needed and snapshot state under entryMu.
	g.entryMu.Lock()
	for insertLevel >= len(g.layers) {
		g.layers = append(g.layers, &layer[K]{nodes: make(map[K]*layerNode[K])})
	}
	// Snapshot layers and entry point — avoids racing on g.layers reads later.
	numLayers := len(g.layers)
	layers := make([]*layer[K], numLayers)
	copy(layers, g.layers)
	var elevator *K
	if g.entryPointKey != nil {
		epCopy := *g.entryPointKey
		elevator = &epCopy
	}
	g.entryMu.Unlock()

	if g.Distance == nil {
		panic("(*Graph).Distance must be set")
	}

	// Traverse from top layer down using snapshot.
	for i := numLayers - 1; i >= 0; i-- {
		ly := layers[i]

		// UPPER LAYERS (above insertLevel): greedy walk, no insertion.
		if i > insertLevel {
			ly.mu.RLock()
			if len(ly.nodes) == 0 {
				ly.mu.RUnlock()
				continue
			}
			searchPoint := ly.arbitraryNode()
			if elevator != nil {
				if ep, ok := ly.nodes[*elevator]; ok {
					searchPoint = ep
				}
			}
			ly.mu.RUnlock()

			// greedyClosest only reads neighbor pointers — lock-free.
			closest := searchPoint.greedyClosest(vec, g.Distance)
			elevator = ptr(closest.Key)
			continue
		}

		// INSERTION LAYERS (at or below insertLevel).
		newNode := &layerNode[K]{
			Node: Node[K]{Key: key, Value: vec},
		}

		ly.mu.Lock()
		if len(ly.nodes) == 0 {
			ly.nodes[key] = newNode
			ly.mu.Unlock()
			continue
		}
		ly.mu.Unlock()

		// Find search start in this layer.
		ly.mu.RLock()
		searchPoint := ly.arbitraryNode()
		if elevator != nil {
			if ep, ok := ly.nodes[*elevator]; ok {
				searchPoint = ep
			}
		}
		ly.mu.RUnlock()

		mLevel := g.mForLevel(i)
		// search reads neighbor pointers — lock-free after initial node lookup.
		neighborhood := searchPoint.search(g.EfSearch, g.EfSearch, vec, g.Distance, nil)
		if len(neighborhood) == 0 {
			panic("no nodes found")
		}
		elevator = ptr(neighborhood[0].node.Key)

		// Select diverse neighbors for the new node.
		candidates := make([]diverseCandidate[K], 0, len(neighborhood))
		for _, n := range neighborhood {
			candidates = append(candidates, diverseCandidate[K]{
				node:        n.node,
				distToOwner: n.dist,
			})
		}
		newNode.storeNeighbors(selectDiverseNeighbors(candidates, mLevel, g.Distance))

		// Insert node into layer map.
		ly.mu.Lock()
		ly.nodes[key] = newNode
		ly.mu.Unlock()

		// Add backlinks from selected neighbors — per-node locking.
		for _, neighbor := range newNode.loadNeighbors() {
			g.nodeLocks.lock(neighbor)
			neighbor.addNeighbor(newNode, mLevel, g.Distance)
			g.nodeLocks.unlock(neighbor)
		}
	}

	// Update entry point if this node has a higher level.
	g.entryMu.Lock()
	if g.entryPointKey == nil || insertLevel > g.entryLevel {
		g.entryPointKey = ptr(key)
		g.entryLevel = insertLevel
	}
	g.entryMu.Unlock()
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
// Safe for concurrent use with AddConcurrent — uses layer RLocks for node map access.
func (h *Graph[K]) SearchWithEf(near Vector, k int, efSearch int, filter FilterFunc[K]) []Node[K] {
	// Snapshot layers and entry point under entryMu.
	h.entryMu.RLock()
	if len(h.layers) == 0 {
		h.entryMu.RUnlock()
		return nil
	}
	numLayers := len(h.layers)
	layers := make([]*layer[K], numLayers)
	copy(layers, h.layers)
	var elevator *K
	if h.entryPointKey != nil {
		epCopy := *h.entryPointKey
		elevator = &epCopy
	}
	h.entryMu.RUnlock()

	for i := numLayers - 1; i >= 0; i-- {
		ly := layers[i]

		// Find search start for this layer under RLock.
		ly.mu.RLock()
		if len(ly.nodes) == 0 {
			ly.mu.RUnlock()
			continue
		}
		searchPoint := ly.arbitraryNode()
		if elevator != nil {
			if ep, ok := ly.nodes[*elevator]; ok {
				searchPoint = ep
			}
		}
		ly.mu.RUnlock()

		// Upper layers: greedy walk to find entry point for next layer.
		if i > 0 {
			closest := searchPoint.greedyClosest(near, h.Distance)
			elevator = ptr(closest.Key)
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

	return nil
}

// Len returns the number of nodes in the graph.
// Safe for concurrent use with AddConcurrent.
func (h *Graph[K]) Len() int {
	h.entryMu.RLock()
	if len(h.layers) == 0 {
		h.entryMu.RUnlock()
		return 0
	}
	base := h.layers[0]
	h.entryMu.RUnlock()

	base.mu.RLock()
	n := len(base.nodes)
	base.mu.RUnlock()
	return n
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
