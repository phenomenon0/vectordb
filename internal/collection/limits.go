package collection

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	// CanonicalMaxSchemaFields keeps one collection from multiplying index and
	// response costs beyond the deliberately small release-candidate surface.
	CanonicalMaxSchemaFields = 8
	// CanonicalMaxVectorDimension is the fixed persisted-schema admission bound.
	CanonicalMaxVectorDimension = 65_536
	// CanonicalMaxSchemaMetadataBytes bounds collection-level metadata after JSON
	// encoding, which is also how the value is persisted in the journal.
	CanonicalMaxSchemaMetadataBytes = 64 << 10
	// CanonicalMaxSearchResponseBytes bounds the estimated in-memory vector copies
	// made while constructing one search response.
	CanonicalMaxSearchResponseBytes = 16 << 20
)

const (
	canonicalSearchResultOverheadBytes       = int64(512)
	canonicalDenseResponseBytesPerDimension  = int64(8)
	canonicalSparseResponseBytesPerDimension = int64(16)
)

var (
	ErrTenantLimitExceeded          = errors.New("canonical tenant limit exceeded")
	ErrCollectionLimitExceeded      = errors.New("canonical collection limit exceeded")
	ErrSearchResponseBudgetExceeded = errors.New("canonical search response budget exceeded")
)

// StoreLimits is immutable deployment admission policy for new durable
// collection creates. Limits are deliberately not persisted: acknowledged
// state must remain replayable if an operator later lowers a configured cap.
type StoreLimits struct {
	MaxTenants     int
	MaxCollections int
}

func (limits StoreLimits) validateRequired() error {
	if limits.MaxTenants <= 0 {
		return fmt.Errorf("max tenants must be positive, got %d", limits.MaxTenants)
	}
	if limits.MaxCollections <= 0 {
		return fmt.Errorf("max collections must be positive, got %d", limits.MaxCollections)
	}
	return nil
}

func validateCanonicalSchemaResourceBounds(schema *CollectionSchema) error {
	if len(schema.Fields) > CanonicalMaxSchemaFields {
		return fmt.Errorf("schema has %d fields; maximum is %d", len(schema.Fields), CanonicalMaxSchemaFields)
	}
	for _, field := range schema.Fields {
		if field.Dim > CanonicalMaxVectorDimension {
			return fmt.Errorf(
				"field %s dimension %d exceeds maximum %d",
				field.Name,
				field.Dim,
				CanonicalMaxVectorDimension,
			)
		}
	}
	metadata, err := json.Marshal(schema.Metadata)
	if err != nil {
		return fmt.Errorf("encode schema metadata: %w", err)
	}
	if len(metadata) > CanonicalMaxSchemaMetadataBytes {
		return fmt.Errorf(
			"schema metadata is %d bytes; maximum is %d",
			len(metadata),
			CanonicalMaxSchemaMetadataBytes,
		)
	}
	return nil
}

func validateCanonicalSearchResponseBudget(schema CollectionSchema, topK int, includeVectors bool) error {
	if !includeVectors || topK <= 0 {
		return nil
	}

	perDocument, err := canonicalSearchResponseBytesPerDocument(schema, true)
	if err != nil {
		return err
	}
	if int64(topK) > int64(CanonicalMaxSearchResponseBytes)/perDocument {
		estimated := perDocument * int64(topK)
		return fmt.Errorf(
			"%w: estimated vector response is %d bytes; maximum is %d",
			ErrSearchResponseBudgetExceeded,
			estimated,
			CanonicalMaxSearchResponseBytes,
		)
	}
	return nil
}

func canonicalSearchResponseBytesPerDocument(schema CollectionSchema, includeVectors bool) (int64, error) {
	perDocument := canonicalSearchResultOverheadBytes
	if !includeVectors {
		return perDocument, nil
	}
	for _, field := range schema.Fields {
		dimension := int64(field.Dim)
		bytesPerDimension := canonicalDenseResponseBytesPerDimension
		if field.Type == VectorTypeSparse {
			bytesPerDimension = canonicalSparseResponseBytesPerDimension
		}
		if dimension > (int64(CanonicalMaxSearchResponseBytes)-perDocument)/bytesPerDimension {
			return 0, fmt.Errorf(
				"%w: one result exceeds the %d-byte budget",
				ErrSearchResponseBudgetExceeded,
				CanonicalMaxSearchResponseBytes,
			)
		}
		perDocument += dimension * bytesPerDimension
	}
	return perDocument, nil
}
