package index

import (
	"sort"
	"sync"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// PayloadIndex maintains secondary indexes on metadata fields for fast filtered search.
// Provides O(1) equality lookups and O(log n) range queries on indexed fields.
type PayloadIndex struct {
	mu         sync.RWMutex
	stringIdx  map[string]map[string]map[uint64]struct{} // field → value → docIDs
	numericIdx map[string]*sortedNumeric                  // field → sorted entries
	fieldDocs  map[string]map[uint64]struct{}             // field → docs that have it (for $exists)
	totalDocs  int
}

// sortedNumeric holds numerically-indexed entries sorted by value for range queries.
type sortedNumeric struct {
	entries []numEntry // sorted by Value
}

type numEntry struct {
	Value float64
	DocID uint64
}

// NewPayloadIndex creates a new PayloadIndex with all maps initialized.
func NewPayloadIndex() *PayloadIndex {
	return &PayloadIndex{
		stringIdx:  make(map[string]map[string]map[uint64]struct{}),
		numericIdx: make(map[string]*sortedNumeric),
		fieldDocs:  make(map[string]map[uint64]struct{}),
	}
}

// toFloat64 converts various numeric types to float64.
// This is a local copy since the filter package's version is unexported.
func payloadToFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int32:
		return float64(val), true
	case int64:
		return float64(val), true
	case uint:
		return float64(val), true
	case uint32:
		return float64(val), true
	case uint64:
		return float64(val), true
	default:
		return 0, false
	}
}

// Index adds a document's metadata fields to the payload index.
func (p *PayloadIndex) Index(id uint64, metadata map[string]interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for field, value := range metadata {
		// Track field existence
		if p.fieldDocs[field] == nil {
			p.fieldDocs[field] = make(map[uint64]struct{})
		}
		p.fieldDocs[field][id] = struct{}{}

		// Index by type
		if strVal, ok := value.(string); ok {
			if p.stringIdx[field] == nil {
				p.stringIdx[field] = make(map[string]map[uint64]struct{})
			}
			if p.stringIdx[field][strVal] == nil {
				p.stringIdx[field][strVal] = make(map[uint64]struct{})
			}
			p.stringIdx[field][strVal][id] = struct{}{}
		} else if numVal, ok := payloadToFloat64(value); ok {
			sn := p.numericIdx[field]
			if sn == nil {
				sn = &sortedNumeric{}
				p.numericIdx[field] = sn
			}
			// Binary search insert to maintain sorted order
			idx := sort.Search(len(sn.entries), func(i int) bool {
				return sn.entries[i].Value >= numVal
			})
			sn.entries = append(sn.entries, numEntry{})
			copy(sn.entries[idx+1:], sn.entries[idx:])
			sn.entries[idx] = numEntry{Value: numVal, DocID: id}
		}
	}
	p.totalDocs++
}

// Remove removes a document's metadata fields from the payload index.
func (p *PayloadIndex) Remove(id uint64, metadata map[string]interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for field, value := range metadata {
		// Remove from fieldDocs
		if docs, ok := p.fieldDocs[field]; ok {
			delete(docs, id)
			if len(docs) == 0 {
				delete(p.fieldDocs, field)
			}
		}

		// Remove from type-specific index
		if strVal, ok := value.(string); ok {
			if valMap, ok := p.stringIdx[field]; ok {
				if docSet, ok := valMap[strVal]; ok {
					delete(docSet, id)
					if len(docSet) == 0 {
						delete(valMap, strVal)
					}
				}
				if len(valMap) == 0 {
					delete(p.stringIdx, field)
				}
			}
		} else if _, ok := payloadToFloat64(value); ok {
			if sn, ok := p.numericIdx[field]; ok {
				for i, entry := range sn.entries {
					if entry.DocID == id {
						sn.entries = append(sn.entries[:i], sn.entries[i+1:]...)
						break
					}
				}
				if len(sn.entries) == 0 {
					delete(p.numericIdx, field)
				}
			}
		}
	}
	p.totalDocs--
}

// QueryBitmap walks the filter tree and returns a candidate set of matching document IDs.
// Returns (nil, false) for unsupported filter operations.
func (p *PayloadIndex) QueryBitmap(f filter.Filter) (map[uint64]struct{}, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.queryFilter(f)
}

func (p *PayloadIndex) queryFilter(f filter.Filter) (map[uint64]struct{}, bool) {
	switch ft := f.(type) {
	case *filter.ComparisonFilter:
		return p.queryComparison(ft)
	case *filter.AndFilter:
		sets := make([]map[uint64]struct{}, 0, len(ft.Filters))
		for _, sub := range ft.Filters {
			s, ok := p.queryFilter(sub)
			if !ok {
				return nil, false
			}
			sets = append(sets, s)
		}
		return queryAnd(sets), true
	case *filter.OrFilter:
		sets := make([]map[uint64]struct{}, 0, len(ft.Filters))
		for _, sub := range ft.Filters {
			s, ok := p.queryFilter(sub)
			if !ok {
				return nil, false
			}
			sets = append(sets, s)
		}
		return queryOr(sets), true
	default:
		// NotFilter, GeoRadiusFilter, GeoBBoxFilter, etc. — unsupported
		return nil, false
	}
}

func (p *PayloadIndex) queryComparison(cf *filter.ComparisonFilter) (map[uint64]struct{}, bool) {
	switch cf.Operator {
	case filter.OpEqual:
		return p.queryEq(cf.Field, cf.Value), true
	case filter.OpNotEqual:
		// ne = all docs with field minus eq matches
		all := p.fieldDocs[cf.Field]
		eq := p.queryEq(cf.Field, cf.Value)
		result := make(map[uint64]struct{}, len(all))
		for id := range all {
			if _, found := eq[id]; !found {
				result[id] = struct{}{}
			}
		}
		return result, true
	case filter.OpIn:
		return p.queryIn(cf.Field, cf.Value), true
	case filter.OpGreaterThan, filter.OpGreaterThanOrEqual, filter.OpLessThan, filter.OpLessThanOrEqual:
		return p.queryRange(cf.Field, cf.Operator, cf.Value)
	case filter.OpExists:
		boolVal, ok := cf.Value.(bool)
		if !ok {
			return nil, false
		}
		docs := p.fieldDocs[cf.Field]
		if boolVal {
			// Return docs that have the field
			result := make(map[uint64]struct{}, len(docs))
			for id := range docs {
				result[id] = struct{}{}
			}
			return result, true
		}
		// exists=false: not directly supportable without full doc set
		return nil, false
	default:
		// regex, contains, startswith, endswith — unsupported
		return nil, false
	}
}

func (p *PayloadIndex) queryEq(field string, value interface{}) map[uint64]struct{} {
	result := make(map[uint64]struct{})

	// Try string lookup
	if strVal, ok := value.(string); ok {
		if valMap, ok := p.stringIdx[field]; ok {
			if docSet, ok := valMap[strVal]; ok {
				for id := range docSet {
					result[id] = struct{}{}
				}
			}
		}
		return result
	}

	// Try numeric lookup
	if numVal, ok := payloadToFloat64(value); ok {
		if sn, ok := p.numericIdx[field]; ok {
			// Binary search for exact match
			idx := sort.Search(len(sn.entries), func(i int) bool {
				return sn.entries[i].Value >= numVal
			})
			for i := idx; i < len(sn.entries) && sn.entries[i].Value == numVal; i++ {
				result[sn.entries[i].DocID] = struct{}{}
			}
		}
		return result
	}

	return result
}

func (p *PayloadIndex) queryIn(field string, values interface{}) map[uint64]struct{} {
	result := make(map[uint64]struct{})

	// Convert values to slice
	var items []interface{}
	switch v := values.(type) {
	case []interface{}:
		items = v
	case []string:
		items = make([]interface{}, len(v))
		for i, s := range v {
			items[i] = s
		}
	case []int:
		items = make([]interface{}, len(v))
		for i, n := range v {
			items[i] = n
		}
	case []float64:
		items = make([]interface{}, len(v))
		for i, n := range v {
			items[i] = n
		}
	default:
		return result
	}

	for _, item := range items {
		eq := p.queryEq(field, item)
		for id := range eq {
			result[id] = struct{}{}
		}
	}
	return result
}

func (p *PayloadIndex) queryRange(field string, op filter.Operator, value interface{}) (map[uint64]struct{}, bool) {
	numVal, ok := payloadToFloat64(value)
	if !ok {
		return nil, false
	}

	sn, ok := p.numericIdx[field]
	if !ok {
		return make(map[uint64]struct{}), true
	}

	result := make(map[uint64]struct{})

	switch op {
	case filter.OpGreaterThan:
		// Find first entry > numVal
		idx := sort.Search(len(sn.entries), func(i int) bool {
			return sn.entries[i].Value > numVal
		})
		for i := idx; i < len(sn.entries); i++ {
			result[sn.entries[i].DocID] = struct{}{}
		}
	case filter.OpGreaterThanOrEqual:
		idx := sort.Search(len(sn.entries), func(i int) bool {
			return sn.entries[i].Value >= numVal
		})
		for i := idx; i < len(sn.entries); i++ {
			result[sn.entries[i].DocID] = struct{}{}
		}
	case filter.OpLessThan:
		idx := sort.Search(len(sn.entries), func(i int) bool {
			return sn.entries[i].Value >= numVal
		})
		for i := 0; i < idx; i++ {
			result[sn.entries[i].DocID] = struct{}{}
		}
	case filter.OpLessThanOrEqual:
		idx := sort.Search(len(sn.entries), func(i int) bool {
			return sn.entries[i].Value > numVal
		})
		for i := 0; i < idx; i++ {
			result[sn.entries[i].DocID] = struct{}{}
		}
	}

	return result, true
}

// queryAnd computes the intersection of multiple sets, starting with the smallest.
func queryAnd(sets []map[uint64]struct{}) map[uint64]struct{} {
	if len(sets) == 0 {
		return make(map[uint64]struct{})
	}

	// Sort by size (smallest first) for early termination
	sort.Slice(sets, func(i, j int) bool {
		return len(sets[i]) < len(sets[j])
	})

	result := make(map[uint64]struct{}, len(sets[0]))
	for id := range sets[0] {
		result[id] = struct{}{}
	}

	for _, s := range sets[1:] {
		for id := range result {
			if _, found := s[id]; !found {
				delete(result, id)
			}
		}
	}

	return result
}

// queryOr computes the union of multiple sets.
func queryOr(sets []map[uint64]struct{}) map[uint64]struct{} {
	totalSize := 0
	for _, s := range sets {
		totalSize += len(s)
	}

	result := make(map[uint64]struct{}, totalSize)
	for _, s := range sets {
		for id := range s {
			result[id] = struct{}{}
		}
	}

	return result
}
