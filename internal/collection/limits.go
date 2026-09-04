package collection

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	// MaxSchemaFields keeps one collection from multiplying index and
	// response costs beyond the deliberately small release-candidate surface.
	MaxSchemaFields = 8
	// MaxVectorDimension is the fixed persisted-schema admission bound.
	MaxVectorDimension = 65_536
	// MaxSchemaMetadataBytes bounds collection-level metadata after JSON
	// encoding, which is also how the value is persisted in the journal.
	MaxSchemaMetadataBytes = 64 << 10
	// MaxSearchResponseBytes bounds the estimated in-memory vector copies
	// made while constructing one search response.
	MaxSearchResponseBytes = 16 << 20
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
	// ErrInvalidSearchArgument marks caller-supplied search parameters that
	// violate the admission contract (score_floor, usage_boost, fallback).
	// Transports map it to a client error (HTTP 400 / gRPC InvalidArgument)
	// rather than a server fault.
	ErrInvalidSearchArgument = errors.New("invalid search argument")
	// ErrInvalidArgument marks any other caller-supplied value the engine
	// rejects (search shape, hybrid params, filters). Same transport mapping.
	ErrInvalidArgument = errors.New("invalid argument")
	// ErrCollectionNotFound, ErrDocumentNotFound, ErrCollectionExists and ErrDocumentExists let
	// transports classify manager errors with errors.Is instead of matching text.
	ErrCollectionNotFound = errors.New("collection not found")
	ErrDocumentNotFound   = errors.New("document not found")
	ErrCollectionExists   = errors.New("collection already exists")
	ErrDocumentExists     = errors.New("document already exists")
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
	if len(schema.Fields) > MaxSchemaFields {
		return fmt.Errorf("schema has %d fields; maximum is %d", len(schema.Fields), MaxSchemaFields)
	}
	for _, field := range schema.Fields {
		if field.Dim > MaxVectorDimension {
			return fmt.Errorf(
				"field %s dimension %d exceeds maximum %d",
				field.Name,
				field.Dim,
				MaxVectorDimension,
			)
		}
	}
	metadata, err := json.Marshal(schema.Metadata)
	if err != nil {
		return fmt.Errorf("encode schema metadata: %w", err)
	}
	if len(metadata) > MaxSchemaMetadataBytes {
		return fmt.Errorf(
			"schema metadata is %d bytes; maximum is %d",
			len(metadata),
			MaxSchemaMetadataBytes,
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
	if int64(topK) > int64(MaxSearchResponseBytes)/perDocument {
		estimated := perDocument * int64(topK)
		return fmt.Errorf(
			"%w: estimated vector response is %d bytes; maximum is %d",
			ErrSearchResponseBudgetExceeded,
			estimated,
			MaxSearchResponseBytes,
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
		if dimension > (int64(MaxSearchResponseBytes)-perDocument)/bytesPerDimension {
			return 0, fmt.Errorf(
				"%w: one result exceeds the %d-byte budget",
				ErrSearchResponseBudgetExceeded,
				MaxSearchResponseBytes,
			)
		}
		perDocument += dimension * bytesPerDimension
	}
	return perDocument, nil
}
