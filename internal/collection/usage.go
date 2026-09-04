package collection

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/phenomenon0/vectordb/internal/hybrid"
)

// usageHalfLife is the decay half-life of a recorded usage signal: after
// this long without further access an entry contributes half of what it
// did when fresh. It is fixed (not configurable) because the tracker is a
// session-length signal, not a ranking subsystem.
const usageHalfLife = time.Hour

// usageEntryCap bounds the tracker so a single collection cannot grow an
// unbounded map from adversarial or churning workloads. When the cap is
// hit, the entry with the lowest current usage score is evicted before a
// new one is inserted.
const usageEntryCap = 250_000

// usageMinScore drops entries whose decayed score has fallen to noise
// level so periodic sweeps stay cheap and the cap only protects against
// genuinely active IDs. time.Duration tops out near 292 years, at which
// a single-hit entry has decayed to ~4e-7, so 1e-6 remains reachable by
// the decay function (a smaller constant would be unprunable).
const usageMinScore = 1e-6

// UsageTracker is an in-memory frecency tracker for one collection.
// It answers "which documents did this tenant actually consume, recently?"
// by recording document IDs that searches returned or GetDocument fetched,
// weighting each hit by count × exponential time decay.
//
// Scope: accreted ranking hint. It is durability class B
// (docs/ARCHITECTURE.md): DurableStore persists it beside the snapshot as
// a usage.json sidecar and reloads it at open, but it is never written to
// the journal or the snapshot itself, and a sidecar that cannot be read is
// discarded loudly rather than failing the store closed. It must therefore
// only ever *nudge* ranking (bounded by UsageBoost), never *replace* the
// similarity signal.
//
// The tracker owns its own mutex and may be used while the caller holds
// Collection.mu in read mode; it must never take Collection.mu.
type UsageTracker struct {
	mu      sync.Mutex
	entries map[uint64]*usageEntry
	now     func() time.Time
}

type usageEntry struct {
	count   uint32
	lastHit time.Time
}

// NewUsageTracker returns an empty tracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{
		entries: make(map[uint64]*usageEntry),
		now:     time.Now,
	}
}

// Record marks a usage hit for docID.
func (t *UsageTracker) Record(docID uint64) {
	if t == nil || docID == 0 {
		return
	}
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()

	entry, ok := t.entries[docID]
	if !ok {
		if len(t.entries) >= usageEntryCap {
			t.evictLowestLocked(now)
		}
		entry = &usageEntry{}
		t.entries[docID] = entry
	}
	entry.count++
	entry.lastHit = now
}

// Score returns the current usage score for docID:
// count / (1 + age/halfLife). Unknown IDs score 0.
func (t *UsageTracker) Score(docID uint64) float64 {
	if t == nil || docID == 0 {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	entry, ok := t.entries[docID]
	if !ok {
		return 0
	}
	age := t.now().Sub(entry.lastHit)
	if age < 0 {
		age = 0
	}
	return float64(entry.count) * decayFactor(age)
}

// Forget drops docID's entry. A deleted document can never be returned
// again, so its frecency is dead weight, and nothing else ever removes an
// entry: it would outlive the document in memory, get written to the
// sidecar, and be restored by every subsequent open. Forgetting also keeps
// the signal honest when an ID is later reused, because the previous
// document's hits must not nudge the new one.
func (t *UsageTracker) Forget(docID uint64) {
	if t == nil || docID == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.entries, docID)
}

// Prune removes entries whose decayed score has fallen to noise level.
// Not required for memory safety — the cap eviction in Record bounds the
// map — but callers that want to release decayed memory early (e.g. an
// ops endpoint) can invoke it on a schedule. Returns the number of
// entries removed.
func (t *UsageTracker) Prune() int {
	if t == nil {
		return 0
	}
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	removed := 0
	for id, entry := range t.entries {
		age := now.Sub(entry.lastHit)
		if age < 0 {
			age = 0
		}
		if float64(entry.count)*decayFactor(age) < usageMinScore {
			delete(t.entries, id)
			removed++
		}
	}
	return removed
}

// Len returns the number of tracked entries.
func (t *UsageTracker) Len() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.entries)
}

// usageDocumentVersion is the schema version of an exported usage
// document. Import accepts no other value: the document is a class B
// ranking hint, so an unrecognized one is discarded loudly instead of
// being guessed at or migrated in place.
const usageDocumentVersion = 1

// UsageRecord is one exported tracker entry. The timestamp is Unix
// milliseconds so the document does not depend on Go's time encoding or
// on the writer's location.
type UsageRecord struct {
	DocID     uint64 `json:"doc_id"`
	Count     uint32 `json:"count"`
	LastHitMS int64  `json:"last_hit_unix_ms"`
}

// UsageDocument is the versioned export of one tracker: exactly the two
// fields Score reads (count and last hit), for every entry.
type UsageDocument struct {
	Version int           `json:"version"`
	Entries []UsageRecord `json:"entries"`
}

// Export returns the tracker's full state, sorted by document ID so that
// identical tracker state always produces identical bytes.
func (t *UsageTracker) Export() UsageDocument {
	doc := UsageDocument{Version: usageDocumentVersion}
	if t == nil {
		return doc
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	doc.Entries = make([]UsageRecord, 0, len(t.entries))
	for id, entry := range t.entries {
		doc.Entries = append(doc.Entries, UsageRecord{
			DocID:     id,
			Count:     entry.count,
			LastHitMS: entry.lastHit.UnixMilli(),
		})
	}
	sort.Slice(doc.Entries, func(a, b int) bool {
		return doc.Entries[a].DocID < doc.Entries[b].DocID
	})
	return doc
}

// Import replaces the tracker's state with doc. It reports an error and
// leaves the tracker untouched when the document is not one this build
// wrote: a wrong version, an entry Record could never have produced, or
// more entries than the cap the tracker enforces at runtime.
func (t *UsageTracker) Import(doc UsageDocument) error {
	if err := validateUsageDocument(doc); err != nil {
		return err
	}
	if t == nil {
		return nil
	}
	entries := make(map[uint64]*usageEntry, len(doc.Entries))
	for _, record := range doc.Entries {
		entries[record.DocID] = &usageEntry{
			count:   record.Count,
			lastHit: time.UnixMilli(record.LastHitMS),
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.entries = entries
	return nil
}

func validateUsageDocument(doc UsageDocument) error {
	if doc.Version != usageDocumentVersion {
		return fmt.Errorf("unsupported usage document version %d", doc.Version)
	}
	if len(doc.Entries) > usageEntryCap {
		return fmt.Errorf("usage document has %d entries, cap is %d", len(doc.Entries), usageEntryCap)
	}
	seen := make(map[uint64]struct{}, len(doc.Entries))
	for _, record := range doc.Entries {
		if record.DocID == 0 || record.Count == 0 {
			return fmt.Errorf("usage document entry for document %d has count %d", record.DocID, record.Count)
		}
		if _, duplicate := seen[record.DocID]; duplicate {
			return fmt.Errorf("usage document repeats document %d", record.DocID)
		}
		seen[record.DocID] = struct{}{}
	}
	return nil
}

// decayFactor returns the harmonic decay 1/(1 + age/halfLife): 1 when
// fresh, 1/2 after one half-life, asymptotic to 0. Harmonic (rather than
// exponential) decay stays representable for arbitrarily old entries
// without underflow.
func decayFactor(age time.Duration) float64 {
	return 1.0 / (1.0 + float64(age)/float64(usageHalfLife))
}

// evictLowestLocked removes the entry with the lowest current score.
// Caller must hold t.mu.
func (t *UsageTracker) evictLowestLocked(now time.Time) {
	var worstID uint64
	var worstScore float64
	first := true
	for id, entry := range t.entries {
		age := now.Sub(entry.lastHit)
		if age < 0 {
			age = 0
		}
		score := float64(entry.count) * decayFactor(age)
		if first || score < worstScore {
			worstID, worstScore, first = id, score, false
		}
	}
	if !first {
		delete(t.entries, worstID)
	}
}

// rankByUsage reorders results by quality × (1 + weight × normalizedUsage),
// where quality is the raw score converted to a higher-is-better scale
// (1/(1+d) for distance fields) and normalizedUsage is the entry's usage
// score divided by the maximum usage score in the set (0 when no entry is
// present, so a first-time result set is untouched). The maximum possible
// boost is (1 + weight), and with no recorded usage at all the order is
// byte-identical to input. Raw scores are returned unchanged; only the
// order is blended.
//
// lowerIsBetter reports whether raw scores are distances (dense fields);
// the multiplicative blend only makes sense on a higher-is-better scale,
// so distances are mapped through 1/(1+d) before blending.
//
// Ties on the blended score keep input order (stable sort), so two
// documents with equal raw score and no usage difference never swap.
func rankByUsage(results []hybrid.SearchResult, usage *UsageTracker, weight float64, lowerIsBetter bool) {
	if usage == nil || weight <= 0 || len(results) < 2 {
		return
	}
	maxUse := 0.0
	uses := make([]float64, len(results))
	for i := range results {
		uses[i] = usage.Score(results[i].DocID)
		if uses[i] > maxUse {
			maxUse = uses[i]
		}
	}
	if maxUse <= 0 {
		return
	}
	type scored struct {
		result hybrid.SearchResult
		order  float64
	}
	merged := make([]scored, len(results))
	for i := range results {
		quality := float64(results[i].Score)
		if lowerIsBetter {
			quality = 1.0 / (1.0 + quality)
		}
		merged[i] = scored{
			result: results[i],
			order:  quality * (1 + weight*uses[i]/maxUse),
		}
	}
	sort.SliceStable(merged, func(a, b int) bool {
		return merged[a].order > merged[b].order
	})
	for i := range merged {
		results[i] = merged[i].result
	}
}
