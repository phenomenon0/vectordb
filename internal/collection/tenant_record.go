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
// field means server default / unlimited. Usage accounting against these
// values is a later step; this release only carries the configured numbers.
type TenantQuota struct {
	MaxDocuments   int64 `json:"max_documents"`
	MaxBytes       int64 `json:"max_bytes"`
	MaxCollections int64 `json:"max_collections"`
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
	// ErrTenantNotFound, ErrTenantExists and ErrTenantSuspended let transports
	// classify tenant lifecycle errors with errors.Is instead of matching text.
	ErrTenantNotFound  = errors.New("tenant not found")
	ErrTenantExists    = errors.New("tenant already exists")
	ErrTenantSuspended = errors.New("tenant is suspended")
)
