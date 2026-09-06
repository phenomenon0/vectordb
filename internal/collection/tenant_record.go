package collection

import (
	"errors"
	"fmt"
)

// Tenant lifecycle status. A tenant record exists independently of whether it
// owns any collections: an administrator can provision or suspend a tenant
// before (or after) its collections come and go.
const (
	TenantStatusActive    = "active"
	TenantStatusSuspended = "suspended"
)

// TenantQuota is administrator-set resource ceilings for one tenant. A zero
// field means server default / unlimited.
type TenantQuota struct {
	MaxDocuments   int64 `json:"max_documents"`
	MaxBytes       int64 `json:"max_bytes"`
	MaxCollections int64 `json:"max_collections"`
}

// TenantUsage is a tenant's current resource consumption, comparable field by
// field against TenantQuota. Collections is always read live from the
// tenant's CollectionManager; Documents and Bytes are the DurableStore's
// running counters.
type TenantUsage struct {
	Documents   int64 `json:"documents"`
	Bytes       int64 `json:"bytes"`
	Collections int64 `json:"collections"`
}

// TenantInfo is the administrator-visible snapshot of one tenant: lifecycle
// status, the quota actually in effect (record override or server default),
// and current usage against it.
type TenantInfo struct {
	TenantID string      `json:"tenant_id"`
	Status   string      `json:"status"`
	Quota    TenantQuota `json:"quota"`
	Usage    TenantUsage `json:"usage"`
}

// TenantRecord is the administrator-visible tenant: its lifecycle status and
// quota, independent of the collections it happens to own.
type TenantRecord struct {
	TenantID string      `json:"tenant_id"`
	Status   string      `json:"status"`
	Quota    TenantQuota `json:"quota"`
}

func (r TenantRecord) validate() error {
	if r.TenantID == "" {
		return errors.New("tenant ID cannot be empty")
	}
	switch r.Status {
	case TenantStatusActive, TenantStatusSuspended:
	default:
		return fmt.Errorf("tenant status must be %q or %q, got %q", TenantStatusActive, TenantStatusSuspended, r.Status)
	}
	if r.Quota.MaxDocuments < 0 || r.Quota.MaxBytes < 0 || r.Quota.MaxCollections < 0 {
		return errors.New("tenant quota fields cannot be negative")
	}
	return nil
}

var (
	// ErrTenantNotFound, ErrTenantExists, ErrTenantSuspended and
	// ErrTenantQuotaExceeded let transports classify tenant lifecycle and
	// admission errors with errors.Is instead of matching text.
	ErrTenantNotFound      = errors.New("tenant not found")
	ErrTenantExists        = errors.New("tenant already exists")
	ErrTenantSuspended     = errors.New("tenant is suspended")
	ErrTenantQuotaExceeded = errors.New("canonical tenant quota exceeded")
)

// documentBytes estimates a document's storage footprint without allocating:
// no JSON marshal, just arithmetic over the shapes already in memory.
// ponytail: estimate, not allocator truth.
func documentBytes(doc *Document) int64 {
	total := int64(8) // ID
	for name, v := range doc.Vectors {
		total += int64(len(name))
		total += 4 * int64(len(v.Dense))
		if v.Sparse != nil {
			total += 8 * int64(len(v.Sparse.Indices))
		}
	}
	total += metadataBytes(doc.Metadata)
	return total
}

// documentsBytes sums documentBytes across a batch.
func documentsBytes(docs []Document) int64 {
	var total int64
	for i := range docs {
		total += documentBytes(&docs[i])
	}
	return total
}

// metadataBytes walks an arbitrary metadata value (string, nested
// map[string]interface{}, []interface{}, nil, or scalar) estimating its
// JSON-serialized size without marshaling it.
func metadataBytes(v interface{}) int64 {
	switch value := v.(type) {
	case nil:
		return 1
	case string:
		return int64(len(value))
	case map[string]interface{}:
		var total int64
		for key, child := range value {
			total += int64(len(key)) + metadataBytes(child)
		}
		return total
	case []interface{}:
		var total int64
		for _, child := range value {
			total += metadataBytes(child)
		}
		return total
	default:
		return 8
	}
}
