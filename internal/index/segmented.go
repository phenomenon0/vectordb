package index

// SegmentedIndex wraps N independent dense-index segments behind the standard
// single-index Index interface, enabling parallel batch builds and fan-out
// search.
//
// Why independent segment graphs scale truly in parallel: a shared HNSW graph
// serializes concurrent inserters on its own lock domains. AddConcurrent
// (internal/index/hnsw/graph.go:651-775) already uses fine-grained locking —
// entryMu for layer growth / entry-point updates, one mutex per layer node
// map, per-node locks for backlink mutation — but every worker still contends
// on the structures of that ONE graph. N disjoint segment graphs have N
// disjoint lock domains, so batch workers build in parallel without sharing
// any lock, and Search fans out across segments on independent RLocks.
//
// Routing: the owning segment of an ID is always id % N (owner below). The
// same function drives Add, Delete, SetMetadata, batch partitioning and
// Import partitioning, so insert/delete/search can never disagree about
// where a document lives — routing is deterministic by construction.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

const (
	// searchOverFetch is the per-segment over-fetch factor for Search: each
	// segment is asked for k*searchOverFetch candidates so that the global
	// merge of top-k is unlikely to be truncated by a segment whose local top-k
	// differs from its contribution to the global top-k. Mirrors HNSW's own 2x
	// adaptive over-fetch for filtered search (hnsw.go:603-616).
	searchOverFetch = 2

	// maxFilterEscalations bounds filtered-search escalation: when a filter
	// leaves merged hits short of k and some segment saturated its fetch
	// limit, only those saturated segments are refetched at double the limit,
	// up to this many times (k*2 -> k*4 -> k*8 -> k*16). Without escalation a
	// selective filter could return fewer than k results even though matching
	// documents exist in other segments.
	maxFilterEscalations = 3
)

// SegmentedIndex implements Index (and BatchAdder, NoCopyBatchAdder and the
// collection-layer metadata setter) over a fixed set of segment indexes.
type SegmentedIndex struct {
	segments []Index

	// n == len(segments), kept as uint64 so owner(id) needs no conversion on
	// the hot routing path.
	n uint64

	// candidatesExamined accumulates per-segment candidate counts across all
	// searches; surfaced via Stats().Extra["candidates_examined"].
	candidatesExamined atomic.Uint64
}

// Compile-time interface conformance checks. HNSWIndex implements BatchAdder
// (hnsw.go:300), NoCopyBatchAdder (hnsw.go:396) and SetMetadata
// (hnsw.go:260), so the wrapper forwards all three; the anonymous struct
// mirrors the inline metadataSetter assertion in
// internal/collection/collection.go:389-395 exactly.
var (
	_ Index            = (*SegmentedIndex)(nil)
	_ BatchAdder       = (*SegmentedIndex)(nil)
	_ NoCopyBatchAdder = (*SegmentedIndex)(nil)
	_ interface {
		SetMetadata(id uint64, metadata map[string]interface{}) error
	} = (*SegmentedIndex)(nil)
)

// NewSegmentedIndex builds a SegmentedIndex with the given number of segments,
// each produced by newSegment. The factory keeps segment construction fully in
// caller hands (e.g. NewHNSWIndex(dim, config) with whatever Params map the
// caller uses today); all segments are expected to share dimension and
// configuration, which Export/Import verify defensively.
//
// Thread-safety per-method as documented; Import must not run concurrently
// with other operations (same contract as Index.Import).
func NewSegmentedIndex(segments int, newSegment func() (Index, error)) (*SegmentedIndex, error) {
	if segments < 1 {
		return nil, fmt.Errorf("segments must be >= 1, got %d", segments)
	}
	if newSegment == nil {
		return nil, errors.New("newSegment factory must not be nil")
	}
	segs := make([]Index, 0, segments)
	for i := 0; i < segments; i++ {
		idx, err := newSegment()
		if err != nil {
			return nil, fmt.Errorf("segment %d: %w", i, err)
		}
		if idx == nil {
			return nil, fmt.Errorf("segment %d: factory returned a nil index", i)
		}
		segs = append(segs, idx)
	}
	return &SegmentedIndex{segments: segs, n: uint64(len(segs))}, nil
}

// SegmentsForNewCollection returns the stable default for schemas that omit
// the segments parameter. The effective topology is durable state: deriving it
// from GOMAXPROCS would make the same journal replay into a different index on
// another host (and would silently multiply graph/search/export concurrency).
// Parallel segmented builds therefore remain explicit and reproducible through
// IndexConfig.Params["segments"].
func SegmentsForNewCollection() int {
	return 1
}

// owner returns the index of the segment owning id. Deterministic everywhere.
func (s *SegmentedIndex) owner(id uint64) int {
	return int(id % s.n)
}

// Name returns "Segmented<segment type>" derived from segment 0, e.g.
// "SegmentedHNSW".
func (s *SegmentedIndex) Name() string {
	return fmt.Sprintf("Segmented%s", s.segments[0].Name())
}

// Add routes to the ID's owning segment; errors propagate verbatim.
func (s *SegmentedIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	return s.segments[s.owner(id)].Add(ctx, id, vector)
}

// Delete routes to the ID's owning segment. The not-found/error contract is
// the owning segment's own (for HNSW: "vector with ID %d not found",
// hnsw.go:702-720). Durable-store paths pre-validate existence upstream, but
// the wrapper deliberately does not mask or reinterpret segment errors.
func (s *SegmentedIndex) Delete(ctx context.Context, id uint64) error {
	return s.segments[s.owner(id)].Delete(ctx, id)
}

// SetMetadata routes to the ID's owning segment. Signature matches the inline
// metadataSetter assertion in internal/collection/collection.go:389-395.
// Segments lacking metadata support are skipped silently, mirroring how the
// collection layer treats such indexes (collection.go:397-399).
func (s *SegmentedIndex) SetMetadata(id uint64, metadata map[string]interface{}) error {
	setter, ok := s.segments[s.owner(id)].(interface {
		SetMetadata(id uint64, metadata map[string]interface{}) error
	})
	if !ok {
		return nil
	}
	return setter.SetMetadata(id, metadata)
}

// Search fans out to ALL segments concurrently, each fetching
// k*searchOverFetch candidates, then merges globally: sort by Distance
// ascending (dense indexes are lower-better; Result.Distance docs say "lower
// = more similar"), ties broken by ascending ID, cut to top-k. CandidatesExamined
// is accumulated across segments (and rounds) into Stats().Extra.
//
// Determinism rule: per-segment results land in fixed slice slots (no append
// races, ascending collection order) and the final comparator is a total order
// (Distance, then ID), so repeated calls return identical order.
//
// Filtered sufficiency: when params carry a filter (detected exactly like
// HNSWIndex.Search does, hnsw.go:557-570) and merged hits < k while some
// segment returned its entire count-capped fetch limit, that segment may have
// been truncated by request size rather than exhaustion — it is refetched at
// double the limit, up to maxFilterEscalations times, until hits >= k, no
// segment remains saturated, or the bound is reached. Unfiltered searches do
// not escalate (deletion-ratio compensation is the segment's own job,
// hnsw.go:617-633).
func (s *SegmentedIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	var filtered bool
	switch p := params.(type) {
	case HNSWSearchParams:
		filtered = p.Filter != nil
	case *HNSWSearchParams:
		filtered = p != nil && p.Filter != nil
	}

	// Segment sizes cap the useful per-segment fetch: HNSW truncates output to
	// the requested k (hnsw.go:683-685) and caps internal fetchK at h.count,
	// so asking a 3-vector segment for 200 candidates can never saturate it
	// and must not trigger escalation.
	counts := make([]int, s.n)
	for i := range s.segments {
		counts[i] = s.segments[i].Stats().Count
	}

	fetch := k * searchOverFetch
	var merged []Result
	perSeg := make([][]Result, s.n) // slot i written only by goroutine i; read after wg.Wait()
	limit := make([]int, s.n)       // effective per-segment fetch limit for the current round
	pending := make([]int, s.n)
	for i := range pending {
		pending[i] = i
	}

	for round := 0; ; round++ {
		var wg sync.WaitGroup
		var mu sync.Mutex
		var errs []error

		for _, i := range pending {
			lim := min(fetch, counts[i])
			limit[i] = lim
			if lim < 1 {
				continue // empty segment: nothing to fetch, never saturates
			}
			wg.Add(1)
			go func(i, lim int) {
				defer wg.Done()
				res, err := s.segments[i].Search(ctx, query, lim, params)
				if err != nil {
					mu.Lock()
					errs = append(errs, fmt.Errorf("segment %d: %w", i, err))
					mu.Unlock()
					return
				}
				perSeg[i] = res
			}(i, lim)
		}
		wg.Wait()
		if len(errs) > 0 {
			return nil, errors.Join(errs...)
		}

		total := 0
		for _, res := range perSeg {
			total += len(res)
		}
		s.candidatesExamined.Add(uint64(total))

		merged = make([]Result, 0, total)
		for _, res := range perSeg { // ascending segment order (determinism contract)
			merged = append(merged, res...)
		}
		sort.Slice(merged, func(a, b int) bool { // total order => deterministic
			if merged[a].Distance != merged[b].Distance {
				return merged[a].Distance < merged[b].Distance
			}
			return merged[a].ID < merged[b].ID
		})

		if len(merged) >= k || !filtered || round >= maxFilterEscalations {
			break
		}

		// Escalate ONLY segments truncated by request size (returned their
		// full limit while holding more than the limit).
		refetch := make([]int, 0, len(pending))
		for _, i := range pending {
			if len(perSeg[i]) >= limit[i] && counts[i] > limit[i] {
				refetch = append(refetch, i)
			}
		}
		if len(refetch) == 0 {
			break
		}
		pending = refetch
		fetch *= 2
	}

	if len(merged) > k {
		merged = merged[:k]
	}
	return merged, nil
}

// BatchAdd partitions the batch by owning segment and inserts each subset into
// its segment concurrently via that segment's BatchAdd (which itself
// parallelizes graph insertion within the subset).
//
// Atomicity mirrors HNSWIndex.BatchAdd's contract (hnsw.go:296-299): on ANY
// segment failure, completed segments roll back this batch's IDs via Delete
// for exactly the IDs they received, and the joined error is returned.
// Two deliberate deviations from HNSW's internal rollback, both consequences
// of compensating through the public Delete API as specified:
//   - rolled-back fresh inserts remain as tombstones (search-invisible,
//     Active counts correct) until Compact/Import reclamation instead of
//     being hard-removed;
//   - compensations use context.WithoutCancel because the incoming context is
//     often itself the failure cause (cancellation) and Delete honors ctx.Err.
//
// Failed segments need no compensation: their own BatchAdd already rolls back
// atomically (hnsw.go:351-361).
func (s *SegmentedIndex) BatchAdd(ctx context.Context, vectors map[uint64][]float32) error {
	return s.batch(ctx, vectors, false)
}

// BatchAddNoCopy behaves identically to BatchAdd but forwards to each
// segment's BatchAddNoCopy. Each slice is owned by exactly one segment, so the
// ownership-transfer contract carries over unchanged (pre-normalization may
// mutate caller slices, hnsw.go:412).
func (s *SegmentedIndex) BatchAddNoCopy(ctx context.Context, vectors map[uint64][]float32) error {
	return s.batch(ctx, vectors, true)
}

func (s *SegmentedIndex) batch(ctx context.Context, vectors map[uint64][]float32, noCopy bool) error {
	parts := make([]map[uint64][]float32, s.n)
	for id, vec := range vectors {
		seg := s.owner(id)
		if parts[seg] == nil {
			parts[seg] = make(map[uint64][]float32, len(vectors)/int(s.n)+1)
		}
		parts[seg][id] = vec
	}

	errs := make([]error, s.n)
	var wg sync.WaitGroup
	for i := range parts {
		if len(parts[i]) == 0 {
			continue
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if noCopy {
				if nc, ok := s.segments[i].(NoCopyBatchAdder); ok {
					errs[i] = nc.BatchAddNoCopy(ctx, parts[i])
					return
				}
			} else if ba, ok := s.segments[i].(BatchAdder); ok {
				errs[i] = ba.BatchAdd(ctx, parts[i])
				return
			}
			// Segment lacks batch support: plain Add loop keeps the wrapper
			// usable with any dense Index implementation.
			for id, vec := range parts[i] {
				if err := s.segments[i].Add(ctx, id, vec); err != nil {
					errs[i] = err
					return
				}
			}
		}(i)
	}
	wg.Wait()

	failed := false
	for _, err := range errs {
		if err != nil {
			failed = true
			break
		}
	}
	if !failed {
		return nil
	}

	rbCtx := context.WithoutCancel(ctx)
	rbErrs := make([]error, 0, len(vectors))
	for i := range parts {
		if errs[i] != nil || len(parts[i]) == 0 {
			continue // failed segment self-rolled back; empty segments got nothing
		}
		for id := range parts[i] {
			if err := s.segments[i].Delete(rbCtx, id); err != nil {
				rbErrs = append(rbErrs, fmt.Errorf("rollback delete %d in segment %d: %w", id, i, err))
			}
		}
	}
	rbErrs = append(rbErrs, errs...) // errors.Join skips nils
	return errors.Join(rbErrs...)
}

// Stats aggregates Count/Deleted/Active/Memory/Disk across segments. Segment
// identity fields (Dim) come from segment 0; Extra carries segmentation
// metrics instead of merging heterogeneous per-segment extras.
func (s *SegmentedIndex) Stats() IndexStats {
	first := s.segments[0].Stats()
	stats := IndexStats{
		Name: fmt.Sprintf("Segmented%s", first.Name),
		Dim:  first.Dim,
		Extra: map[string]interface{}{
			"segments":            len(s.segments),
			"candidates_examined": s.candidatesExamined.Load(),
		},
	}
	for _, seg := range s.segments {
		st := seg.Stats()
		stats.Count += st.Count
		stats.Deleted += st.Deleted
		stats.Active += st.Active
		stats.MemoryUsed += st.MemoryUsed
		stats.DiskUsed += st.DiskUsed
	}
	return stats
}

// segmentedVectorEntry and segmentedExportFormat mirror the local export types
// of HNSWIndex.Export (hnsw.go:797-812) field-for-field, including JSON tag
// order: encoding/json emits struct fields in declaration order, which matters
// because Export must produce byte-identical output to a single-index export
// of the same document set. quantizerState (quantization.go:37-45) lives in
// this package and is reused directly.
type segmentedVectorEntry struct {
	ID        uint64                 `json:"id"`
	Vector    []float32              `json:"vector,omitempty"`
	Quantized []byte                 `json:"quantized,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Deleted   bool                   `json:"deleted,omitempty"`
}

type segmentedExportFormat struct {
	Version   int                    `json:"version"`
	Dim       int                    `json:"dim"`
	Config    map[string]interface{} `json:"config"`
	Quantizer *quantizerState        `json:"quantizer,omitempty"`
	Vectors   []segmentedVectorEntry `json:"vectors"`
	Deleted   []uint64               `json:"deleted,omitempty"`
}

// Export merges all segment exports into a single v2-format snapshot that is
// byte-identical to what ONE HNSWIndex containing the same document set would
// emit (vectors sorted by ID, config/dim/quantizer from segment 0).
//
// Byte-equality caveat: with tombstones present, the reference single-index
// export emits its top-level "deleted" array in Go map-iteration order
// (hnsw.go:839-842), which is nondeterministic even between two exports of
// the SAME plain index. This implementation therefore sorts the merged deleted
// array — byte-identity holds whenever there are no tombstones, which is the
// steady state after Import since v2 import reclaims them (hnsw.go:936-949).
func (s *SegmentedIndex) Export() ([]byte, error) {
	parsed := make([]*segmentedExportFormat, s.n)
	for i, seg := range s.segments {
		blob, err := seg.Export()
		if err != nil {
			return nil, fmt.Errorf("segment %d export: %w", i, err)
		}
		var ex segmentedExportFormat
		if err := json.Unmarshal(blob, &ex); err != nil {
			return nil, fmt.Errorf("segment %d export parse: %w", i, err)
		}
		if ex.Version != 2 {
			return nil, fmt.Errorf("segment %d: unsupported export version %d", i, ex.Version)
		}
		if i > 0 {
			if ex.Dim != parsed[0].Dim {
				return nil, fmt.Errorf("segment %d dim %d != segment 0 dim %d", i, ex.Dim, parsed[0].Dim)
			}
			cfgA, _ := json.Marshal(parsed[0].Config)
			cfgB, _ := json.Marshal(ex.Config)
			if !bytes.Equal(cfgA, cfgB) {
				return nil, fmt.Errorf("segment %d config differs from segment 0", i)
			}
			qA, _ := json.Marshal(parsed[0].Quantizer)
			qB, _ := json.Marshal(ex.Quantizer)
			if !bytes.Equal(qA, qB) {
				return nil, fmt.Errorf("segment %d quantizer state differs from segment 0", i)
			}
		}
		parsed[i] = &ex
	}

	seen := make(map[uint64]struct{})
	vectors := make([]segmentedVectorEntry, 0)
	deletedSet := make(map[uint64]struct{})
	for _, ex := range parsed { // ascending segment order
		for _, e := range ex.Vectors {
			if _, dup := seen[e.ID]; dup {
				return nil, fmt.Errorf("duplicate ID %d across segments", e.ID) // unreachable by routing; defensive
			}
			seen[e.ID] = struct{}{}
			vectors = append(vectors, e)
		}
		for _, id := range ex.Deleted {
			deletedSet[id] = struct{}{}
		}
	}
	sort.Slice(vectors, func(a, b int) bool { return vectors[a].ID < vectors[b].ID }) // unique IDs => total order

	deleted := make([]uint64, 0, len(deletedSet))
	for id := range deletedSet {
		deleted = append(deleted, id)
	}
	sort.Slice(deleted, func(a, b int) bool { return deleted[a] < deleted[b] })

	out := segmentedExportFormat{
		Version:   parsed[0].Version,
		Dim:       parsed[0].Dim,
		Config:    parsed[0].Config,
		Quantizer: parsed[0].Quantizer,
		Vectors:   vectors,
		Deleted:   deleted,
	}
	return json.Marshal(out)
}

// Import parses one snapshot, validates version/dimension, partitions entries
// and tombstone IDs by owner(id), and feeds each segment its subset through
// that segment's own Import. Rebuilding per segment also reclaims tombstones
// for free on v2 data (hnsw.go:936-949). v1 snapshots keep their semantics:
// owned Deleted-ID arrays are passed through so segments re-tombstone them.
//
// Segments rebuild concurrently (disjoint graphs, no shared locks). If any
// segment fails, the joined error is returned; earlier-completed segments keep
// their rebuilt state — pre-validation here covers everything that can be
// checked centrally (version, dim), mirroring how a plain HNSW import can fail
// partway through entry processing.
func (s *SegmentedIndex) Import(data []byte) error {
	var imp segmentedExportFormat
	if err := json.Unmarshal(data, &imp); err != nil {
		return fmt.Errorf("failed to unmarshal segmented index: %w", err)
	}
	if imp.Version != 1 && imp.Version != 2 {
		return fmt.Errorf("unsupported HNSW export version: %d", imp.Version)
	}
	dim := s.segments[0].Stats().Dim
	if imp.Dim != dim {
		return fmt.Errorf("dimension mismatch: index is %d, import data is %d", dim, imp.Dim)
	}

	parts := make([]*segmentedExportFormat, s.n)
	for i := range parts {
		parts[i] = &segmentedExportFormat{
			Version:   imp.Version,
			Dim:       imp.Dim,
			Config:    imp.Config,
			Quantizer: imp.Quantizer,
		}
	}
	for _, e := range imp.Vectors {
		seg := s.owner(e.ID)
		parts[seg].Vectors = append(parts[seg].Vectors, e)
	}
	for _, id := range imp.Deleted {
		seg := s.owner(id)
		parts[seg].Deleted = append(parts[seg].Deleted, id)
	}

	errs := make([]error, s.n)
	var wg sync.WaitGroup
	for i := range parts {
		blob, err := json.Marshal(parts[i])
		if err != nil {
			return fmt.Errorf("partition %d marshal: %w", i, err)
		}
		wg.Add(1)
		go func(i int, blob []byte) {
			defer wg.Done()
			errs[i] = s.segments[i].Import(blob)
		}(i, blob)
	}
	wg.Wait()
	return errors.Join(errs...)
}
