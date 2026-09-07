package main

import (
	"github.com/phenomenon0/vectordb/api/contract"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/filter"
	"github.com/phenomenon0/vectordb/internal/releaseinfo"
)

// canonicalFilterOperators is the comparison operator set filter.FromMap
// accepts, spelled the way a caller writes it in a filter object ("$" plus
// the operator) and named by its constant so a rename breaks the build.
//
// ponytail: the parser has no registry, so a newly added operator has to be
// added here too; the drift test in contract_test.go checks each one against
// the recall schema's filter description.
var canonicalFilterOperators = []string{
	"$" + string(filter.OpEqual),
	"$" + string(filter.OpNotEqual),
	"$" + string(filter.OpGreaterThan),
	"$" + string(filter.OpGreaterThanOrEqual),
	"$" + string(filter.OpLessThan),
	"$" + string(filter.OpLessThanOrEqual),
	"$" + string(filter.OpIn),
	"$" + string(filter.OpNotIn),
	"$" + string(filter.OpContains),
	"$" + string(filter.OpStartsWith),
	"$" + string(filter.OpEndsWith),
	"$" + string(filter.OpRegex),
	"$" + string(filter.OpExists),
	"$" + string(filter.OpGeoRadius),
	"$" + string(filter.OpGeoBBox),
}

// statusPayload is the GET /v3/status body: what this build is, what it
// serves, what it will embed with, and where it says no. Every value is
// derived — the operation list from api/contract/v3/operations.json, the
// limits from the Max* consts and the rate-limit environment, the
// embedding block from the process embedder — so nothing here can drift
// from the server that answers.
func statusPayload(embedder *serverEmbedder, limits limitConfig, usageLoaded, readOnly bool, faultedTenants []string, requestID string) (map[string]any, error) {
	ops, err := contract.Operations()
	if err != nil {
		return nil, err
	}
	httpOps := make([]map[string]string, 0, len(ops))
	grpcRPCs := make([]string, 0, len(ops))
	mcpTools := make([]string, 0, len(ops))
	seenRPC := map[string]bool{}
	seenTool := map[string]bool{}
	for _, op := range ops {
		httpOps = append(httpOps, map[string]string{
			"method":     op.Method,
			"path":       op.Path,
			"permission": op.Permission,
		})
		if op.GRPCRPC != "" && !seenRPC[op.GRPCRPC] {
			seenRPC[op.GRPCRPC] = true
			grpcRPCs = append(grpcRPCs, op.GRPCRPC)
		}
		if op.MCPTool != "" && !seenTool[op.MCPTool] {
			seenTool[op.MCPTool] = true
			mcpTools = append(mcpTools, op.MCPTool)
		}
	}

	embedding := map[string]any{
		"provider":  "none",
		"model":     "",
		"dim":       0,
		"available": false,
	}
	if embedder != nil {
		embedding = map[string]any{
			"provider":  embedder.Provider,
			"model":     embedder.Model,
			"dim":       embedder.Dim(),
			"available": true,
		}
	}

	return map[string]any{
		"version": releaseinfo.Version(),
		"contract": map[string]any{
			"http": httpOps,
			"grpc": grpcRPCs,
			"mcp":  mcpTools,
		},
		"embedding": embedding,
		"limits": map[string]any{
			"max_schema_fields":         vcollection.MaxSchemaFields,
			"max_schema_metadata_bytes": vcollection.MaxSchemaMetadataBytes,
			"max_vector_dimension":      vcollection.MaxVectorDimension,
			"max_search_fields":         vcollection.MaxSearchFields,
			"max_search_top_k":          vcollection.MaxSearchTopK,
			"max_search_ef":             vcollection.MaxSearchEf,
			"max_search_response_bytes": vcollection.MaxSearchResponseBytes,
			"max_batch_documents":       vcollection.MaxBatchDocuments,
			"api_rps":                   limits.APIRPS,
			"tenant_rps":                limits.TenantRPS,
			"tenant_burst":              limits.TenantBurst,
			"auth_failure_rps":          limits.AuthFailureRPS,
			"auth_failure_burst":        limits.AuthFailureBurst,
			"max_rate_limit_keys":       limits.MaxRateLimitKeys,
			"max_tenants":               limits.MaxTenants,
			"max_collections":           limits.MaxCollections,
			"max_tenant_documents":      limits.MaxTenantDocuments,
			"max_tenant_bytes":          limits.MaxTenantBytes,
			"max_tenant_collections":    limits.MaxTenantCollections,
		},
		"capabilities": map[string]any{
			"texts":       embedder != nil,
			"filters":     canonicalFilterOperators,
			"hybrid":      []string{"rrf", "weighted", "linear"},
			"fallback":    true,
			"usage_boost": true,
			"index_types": vcollection.IndexTypeNames(),
			// The durability classes a create-collection request may ask for
			// (ADR 0009); an ephemeral collection's documents are memory only.
			"durability_classes": []string{vcollection.DurabilityDurable, vcollection.DurabilityEphemeral},
			// read_only is true when this node serves a read replica
			// directory: every operation below with a write or admin
			// permission is refused with permission_denied here, whatever
			// claim the caller's token carries. It is the same fact /readyz
			// reports under the same name.
			"read_only": readOnly,
		},
		// signals is the accreted-signal surface (durability class B).
		// usage.loaded is false only when a usage sidecar existed and was
		// discarded, so an operator reading false knows ranking hints were
		// lost and searches answer by similarity alone until they accrete
		// again (CTL-05). tenants.faulted names tenants isolated by a
		// per-tenant open fault (see /readyz's faulted_tenants); the process
		// itself is still ready and serves every other tenant.
		"signals": map[string]any{
			"usage":   map[string]any{"loaded": usageLoaded},
			"tenants": map[string]any{"faulted": faultedTenants},
		},
		"request_id": requestID,
	}, nil
}
