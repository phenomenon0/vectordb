package main

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/security"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/known/structpb"
)

func canonicalGRPCAdminContext(tenantID string) context.Context {
	return context.WithValue(context.Background(), security.TenantContextKey, &security.TenantContext{
		TenantID: tenantID,
		Permissions: map[string]bool{
			"read":  true,
			"write": true,
			"admin": true,
		},
		Collections: make(map[string]bool),
		IsAdmin:     true,
	})
}

// canonicalGRPCServerAdminContext mimics the context a static server-admin
// credential produces, the same way canonicalGRPCAdminContext mimics a
// tenant-scoped one.
func canonicalGRPCServerAdminContext() context.Context {
	return context.WithValue(context.Background(), security.TenantContextKey, &security.TenantContext{
		IsServerAdmin: true,
	})
}

func canonicalGRPCScopedContext(tenantID string, permissions map[string]bool, collections ...string) context.Context {
	allowed := make(map[string]bool, len(collections))
	for _, collection := range collections {
		allowed[collection] = true
	}
	return context.WithValue(context.Background(), security.TenantContextKey, &security.TenantContext{
		TenantID:    tenantID,
		Permissions: permissions,
		Collections: allowed,
	})
}

func mustProtoStruct(t *testing.T, values map[string]interface{}) *structpb.Struct {
	t.Helper()
	result, err := structpb.NewStruct(values)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func denseProtoVector(values ...float32) *deepdatav3.VectorData {
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Dense{
		Dense: &deepdatav3.DenseVector{Values: values},
	}}
}

func sparseProtoVector(dim int32, indices []uint32, values []float32) *deepdatav3.VectorData {
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Sparse{
		Sparse: &deepdatav3.SparseVector{Dim: dim, Indices: indices, Values: values},
	}}
}

func TestCanonicalGRPCMirrorsTenantCollectionMutationAndHybridSearchContract(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close durable store: %v", err)
		}
	})

	server := &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}
	ctx := canonicalGRPCAdminContext("acme")
	collectionMetadata := mustProtoStruct(t, map[string]interface{}{
		"owner":  "retrieval",
		"policy": map[string]interface{}{"tier": float64(2), "active": true},
	})

	created, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId:    "acme",
		Name:        "docs",
		Description: "canonical hybrid documents",
		Metadata:    collectionMetadata,
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 2, IndexType: "hnsw"},
			{Name: "keywords", Type: int32(vcollection.VectorTypeSparse), Dim: 8, IndexType: "inverted"},
			{
				Name:        "flat_aux",
				Type:        int32(vcollection.VectorTypeDense),
				Dim:         2,
				IndexType:   "flat",
				IndexParams: mustProtoStruct(t, map[string]interface{}{"metric": "euclidean"}),
			},
		},
	})
	if err != nil || created.Name != "docs" {
		t.Fatalf("create collection: response=%+v err=%v", created, err)
	}

	inserted, err := server.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Id:         7,
		Vectors: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(1, 0),
			"keywords":  sparseProtoVector(8, []uint32{1, 3}, []float32{2, 1}),
			"flat_aux":  denseProtoVector(1, 0),
		},
		Metadata: mustProtoStruct(t, map[string]interface{}{
			"kind":   "alpha",
			"nested": map[string]interface{}{"rank": float64(1), "verified": true},
		}),
	})
	if err != nil || inserted.Id != 7 {
		t.Fatalf("insert: response=%+v err=%v", inserted, err)
	}

	batch, err := server.BatchInsert(ctx, &deepdatav3.BatchInsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Docs: []*deepdatav3.BatchDoc{{
			Vectors: map[string]*deepdatav3.VectorData{
				"embedding": denseProtoVector(0, 1),
				"keywords":  sparseProtoVector(8, []uint32{2}, []float32{3}),
				"flat_aux":  denseProtoVector(0, 1),
			},
			Metadata: mustProtoStruct(t, map[string]interface{}{"kind": "beta"}),
		}},
	})
	if err != nil || batch.Inserted != 1 || len(batch.Ids) != 1 || batch.Ids[0] != 8 {
		t.Fatalf("batch insert: response=%+v err=%v", batch, err)
	}

	// Abort without a checkpoint to model a process crash after both gRPC
	// acknowledgements. Reopening must recover the collection and documents
	// from the canonical journal, not from graceful-shutdown persistence.
	if err := store.Abort(); err != nil {
		t.Fatalf("abort durable store: %v", err)
	}
	reopened, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen durable store after acknowledged gRPC mutations: %v", err)
	}
	store = reopened
	server = &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}

	tenantInfo, err := server.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"})
	if err != nil || tenantInfo.CollectionCount != 1 || tenantInfo.TotalDocuments != 2 || len(tenantInfo.Collections) != 1 {
		t.Fatalf("tenant info: response=%+v err=%v", tenantInfo, err)
	}
	listed, err := server.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: "acme"})
	if err != nil || len(listed.Collections) != 1 || listed.Collections[0].Name != "docs" {
		t.Fatalf("list collections: response=%+v err=%v", listed, err)
	}
	got, err := server.GetCollection(ctx, &deepdatav3.GetCollectionRequest{TenantId: "acme", Name: "docs"})
	if err != nil || got.Collection == nil {
		t.Fatalf("get collection: response=%+v err=%v", got, err)
	}
	if got.Collection.Description != "canonical hybrid documents" || got.Collection.DocumentCount != 2 || len(got.Collection.Fields) != 3 {
		t.Fatalf("get collection lost schema/count: %+v", got.Collection)
	}
	if got.Collection.Metadata.AsMap()["owner"] != "retrieval" {
		t.Fatalf("get collection lost Struct metadata: %v", got.Collection.Metadata)
	}
	var flatParams *structpb.Struct
	for _, field := range got.Collection.Fields {
		if field.Name == "flat_aux" {
			flatParams = field.IndexParams
			break
		}
	}
	if flatParams == nil || flatParams.AsMap()["metric"] != "euclidean" {
		t.Fatalf("get collection lost string-valued Flat index params: %v", flatParams)
	}

	searched, err := server.Search(ctx, &deepdatav3.SearchRequest{
		TenantId:   "acme",
		Collection: "docs",
		TopK:       2,
		Queries: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(1, 0),
			"keywords":  sparseProtoVector(8, []uint32{1}, []float32{2}),
		},
		IncludeVectors: true,
		Filters: mustProtoStruct(t, map[string]interface{}{
			"kind": map[string]interface{}{"$eq": "alpha"},
		}),
		HybridParams: &deepdatav3.HybridSearchParams{
			Strategy:    "weighted",
			Weights:     map[string]float32{"embedding": 0.7, "keywords": 0.3},
			RrfConstant: 42,
		},
	})
	if err != nil || len(searched.Results) == 0 {
		t.Fatalf("hybrid search: response=%+v err=%v", searched, err)
	}
	hit := searched.Results[0]
	if hit.Id != 7 || hit.Metadata.AsMap()["kind"] != "alpha" {
		t.Fatalf("hybrid search returned wrong typed hit: %+v", hit)
	}
	if dense := hit.Vectors["embedding"].GetDense(); dense == nil || len(dense.Values) != 2 || dense.Values[0] != 1 {
		t.Fatalf("dense result vector was not returned: %+v", hit.Vectors["embedding"])
	}
	if sparse := hit.Vectors["keywords"].GetSparse(); sparse == nil || sparse.Dim != 8 || len(sparse.Indices) != 2 || sparse.Indices[0] != 1 {
		t.Fatalf("sparse result vector was not returned: %+v", hit.Vectors["keywords"])
	}
	if flat := hit.Vectors["flat_aux"].GetDense(); flat == nil || len(flat.Values) != 2 || flat.Values[0] != 1 {
		t.Fatalf("auxiliary Flat result vector was not returned: %+v", hit.Vectors["flat_aux"])
	}

	if _, err := server.DeleteDoc(ctx, &deepdatav3.DeleteDocRequest{TenantId: "acme", Collection: "docs", DocId: 8}); err != nil {
		t.Fatalf("delete document: %v", err)
	}
	if _, err := server.DeleteCollection(ctx, &deepdatav3.DeleteCollectionRequest{TenantId: "acme", Name: "docs"}); err != nil {
		t.Fatalf("delete collection: %v", err)
	}
	tenantInfo, err = server.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"})
	if err != nil || tenantInfo.CollectionCount != 0 || tenantInfo.TotalDocuments != 0 {
		t.Fatalf("tenant info after deletes: response=%+v err=%v", tenantInfo, err)
	}
}

func TestCanonicalGRPCUpsertAndGetDocReplaceAndReplay(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close durable store: %v", err)
		}
	})

	server := &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}
	ctx := canonicalGRPCAdminContext("acme")
	if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId:    "acme",
		Name:        "docs",
		Description: "upsert documents",
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 2, IndexType: "hnsw"},
		},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	upserted, err := server.Upsert(ctx, &deepdatav3.UpsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Id:         22,
		Vectors: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(1, 0),
		},
		Metadata: mustProtoStruct(t, map[string]interface{}{"kind": "first"}),
	})
	if err != nil || upserted.Id != 22 {
		t.Fatalf("first upsert: response=%+v err=%v", upserted, err)
	}

	// Replacement keeps a single live document and resolves the new vector.
	upserted, err = server.Upsert(ctx, &deepdatav3.UpsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Id:         22,
		Vectors: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(1, 1),
		},
		Metadata: mustProtoStruct(t, map[string]interface{}{"kind": "second"}),
	})
	if err != nil || upserted.Id != 22 {
		t.Fatalf("replacement upsert: response=%+v err=%v", upserted, err)
	}

	got, err := server.GetDoc(ctx, &deepdatav3.GetDocRequest{TenantId: "acme", Collection: "docs", DocId: 22})
	if err != nil {
		t.Fatalf("get doc: %v", err)
	}
	if got.Id != 22 || got.Metadata.AsMap()["kind"] != "second" {
		t.Fatalf("get after upsert returned %+v", got)
	}
	if dense := got.Vectors["embedding"].GetDense(); dense == nil || len(dense.Values) != 2 || dense.Values[0] != 1 || dense.Values[1] != 1 {
		t.Fatalf("get lost replaced vector: %+v", got.Vectors)
	}

	if _, err := server.GetDoc(ctx, &deepdatav3.GetDocRequest{TenantId: "acme", Collection: "docs", DocId: 99}); err == nil {
		t.Fatal("get of missing doc should fail")
	}

	// Abort models a crash; replay must surface the upsert document, not the
	// original vector.
	if err := store.Abort(); err != nil {
		t.Fatalf("abort durable store: %v", err)
	}
	reopened, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen durable store after upsert: %v", err)
	}
	store = reopened
	server = &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}
	got, err = server.GetDoc(ctx, &deepdatav3.GetDocRequest{TenantId: "acme", Collection: "docs", DocId: 22})
	if err != nil {
		t.Fatalf("get doc after replay: %v", err)
	}
	if got.Metadata.AsMap()["kind"] != "second" {
		t.Fatalf("upsert did not replay as replacement: %+v", got)
	}
	tenantInfo, err := server.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"})
	if err != nil || tenantInfo.TotalDocuments != 1 {
		t.Fatalf("tenant info after upsert replay: response=%+v err=%v", tenantInfo, err)
	}
}

func TestCanonicalGRPCPersistenceHealthGatesEveryRPC(t *testing.T) {
	fault := errors.New("journal append state is indeterminate")
	healthCalls := 0
	server := &CollectionGRPCServer{
		tenants: vcollection.NewTenantManager(""),
		persistenceHealth: func() error {
			healthCalls++
			return fault
		},
	}
	ctx := canonicalGRPCAdminContext("acme")
	tests := []struct {
		name string
		call func() error
	}{
		{"get tenant info", func() error { _, err := server.GetTenantInfo(ctx, nil); return err }},
		{"list collections", func() error { _, err := server.ListCollections(ctx, nil); return err }},
		{"get collection", func() error { _, err := server.GetCollection(ctx, nil); return err }},
		{"create collection", func() error { _, err := server.CreateCollection(ctx, nil); return err }},
		{"delete collection", func() error { _, err := server.DeleteCollection(ctx, nil); return err }},
		{"insert", func() error { _, err := server.Insert(ctx, nil); return err }},
		{"batch insert", func() error { _, err := server.BatchInsert(ctx, nil); return err }},
		{"search", func() error { _, err := server.Search(ctx, nil); return err }},
		{"delete doc", func() error { _, err := server.DeleteDoc(ctx, nil); return err }},
		{"upsert", func() error { _, err := server.Upsert(ctx, nil); return err }},
		{"get doc", func() error { _, err := server.GetDoc(ctx, nil); return err }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if code := status.Code(test.call()); code != codes.Unavailable {
				t.Fatalf("persistence fault code = %s, want Unavailable", code)
			}
		})
	}
	if healthCalls != len(tests) {
		t.Fatalf("health callback calls = %d, want %d", healthCalls, len(tests))
	}

	withoutHealth := &CollectionGRPCServer{tenants: vcollection.NewTenantManager("")}
	if _, err := withoutHealth.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"}); status.Code(err) != codes.Unavailable {
		t.Fatalf("missing health callback error = %v, want Unavailable", err)
	}
}

func TestCanonicalGRPCListReadsPropagateDurableClosureAfterHealthCheck(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	server := &CollectionGRPCServer{
		tenants:           store.Tenants(),
		persistenceHealth: func() error { return nil }, // Model a stale pre-RPC health result.
	}
	if err := store.Abort(); err != nil {
		t.Fatalf("abort durable store: %v", err)
	}

	ctx := canonicalGRPCAdminContext("acme")
	calls := []struct {
		name string
		call func() error
	}{
		{"get tenant info", func() error {
			_, err := server.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"})
			return err
		}},
		{"list collections", func() error {
			_, err := server.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: "acme"})
			return err
		}},
	}
	for _, call := range calls {
		t.Run(call.name, func(t *testing.T) {
			if code := status.Code(call.call()); code != codes.Unavailable {
				t.Fatalf("closed durable store code = %s, want Unavailable", code)
			}
		})
	}
}

func TestCanonicalGRPCMutationAdmissionProtectsJournalBoundary(t *testing.T) {
	server := &CollectionGRPCServer{
		tenants:           vcollection.NewTenantManager(""),
		persistenceHealth: func() error { return nil },
	}
	ctx := canonicalGRPCAdminContext("acme")
	if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme",
		Name:     "not/addressable",
		Fields: []*deepdatav3.VectorFieldConfig{{
			Name: "dense", Type: int32(vcollection.VectorTypeDense), Dim: 1, IndexType: "flat",
		}},
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("invalid collection name code = %s, want InvalidArgument (err=%v)", status.Code(err), err)
	}
	oversizedMetadata := mustProtoStruct(t, map[string]interface{}{
		"blob": strings.Repeat("x", canonicalGRPCMutationMaxProtoBytes+1024),
	})

	if _, err := server.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Vectors:    map[string]*deepdatav3.VectorData{"dense": denseProtoVector(1)},
		Metadata:   oversizedMetadata,
	}); status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("oversized insert code = %s, want ResourceExhausted (err=%v)", status.Code(err), err)
	}

	if _, err := server.BatchInsert(ctx, &deepdatav3.BatchInsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Docs: []*deepdatav3.BatchDoc{{
			Vectors:  map[string]*deepdatav3.VectorData{"dense": denseProtoVector(1)},
			Metadata: oversizedMetadata,
		}},
	}); status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("oversized batch code = %s, want ResourceExhausted (err=%v)", status.Code(err), err)
	}

	t.Run("batch count", func(t *testing.T) {
		tooMany := make([]*deepdatav3.BatchDoc, vcollection.MaxBatchDocuments+1)
		if _, err := server.BatchInsert(ctx, &deepdatav3.BatchInsertRequest{
			TenantId:   "acme",
			Collection: "docs",
			Docs:       tooMany,
		}); status.Code(err) != codes.ResourceExhausted {
			t.Fatalf("oversized batch count code = %s, want ResourceExhausted (err=%v)", status.Code(err), err)
		}
	})
}

func TestCanonicalGRPCSearchAdmissionIsBounded(t *testing.T) {
	server := &CollectionGRPCServer{
		tenants:           vcollection.NewTenantManager(""),
		persistenceHealth: func() error { return nil },
	}
	ctx := canonicalGRPCAdminContext("acme")
	// The bounds are checked by the engine, which needs the collection to
	// exist; a missing collection is NotFound and would mask the check.
	if _, err := server.CreateCollection(ctx, canonicalGRPCCreateCollectionRequest("acme", "docs")); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	if _, err := server.Search(ctx, &deepdatav3.SearchRequest{
		TenantId:   "acme",
		Collection: "docs",
		TopK:       int32(vcollection.MaxSearchTopK + 1),
		Queries:    map[string]*deepdatav3.VectorData{"dense": denseProtoVector(1)},
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("oversized top_k code = %s, want InvalidArgument (err=%v)", status.Code(err), err)
	}

	queries := make(map[string]*deepdatav3.VectorData, vcollection.MaxSearchFields+1)
	for i := 0; i <= vcollection.MaxSearchFields; i++ {
		queries[fmt.Sprintf("field-%d", i)] = denseProtoVector(1)
	}
	if _, err := server.Search(ctx, &deepdatav3.SearchRequest{
		TenantId:     "acme",
		Collection:   "docs",
		TopK:         1,
		Queries:      queries,
		HybridParams: &deepdatav3.HybridSearchParams{Strategy: "rrf"},
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("too many query fields code = %s, want InvalidArgument (err=%v)", status.Code(err), err)
	}
}

func TestCanonicalGRPCPreservesTenantPermissionAndCollectionScope(t *testing.T) {
	server := &CollectionGRPCServer{
		tenants:           vcollection.NewTenantManager(""),
		persistenceHealth: func() error { return nil },
	}
	readOnly := canonicalGRPCScopedContext("acme", map[string]bool{"read": true}, "docs")
	if _, err := server.GetCollection(readOnly, &deepdatav3.GetCollectionRequest{TenantId: "other", Name: "docs"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("cross-tenant get code = %s, want PermissionDenied", status.Code(err))
	}
	if _, err := server.GetCollection(readOnly, &deepdatav3.GetCollectionRequest{TenantId: "acme", Name: "secret"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("out-of-scope get code = %s, want PermissionDenied", status.Code(err))
	}
	if _, err := server.Insert(readOnly, &deepdatav3.InsertRequest{TenantId: "acme", Collection: "docs"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("read-only insert code = %s, want PermissionDenied", status.Code(err))
	}
	if _, err := server.ListCollections(readOnly, &deepdatav3.ListCollectionsRequest{TenantId: "acme"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("non-admin list code = %s, want PermissionDenied", status.Code(err))
	}

	tenantAdmin := canonicalGRPCScopedContext("acme", map[string]bool{"admin": true}, "docs")
	tenantContext, _ := security.GetTenantContextFromContext(tenantAdmin)
	tenantContext.IsAdmin = true
	if _, err := server.GetCollection(tenantAdmin, &deepdatav3.GetCollectionRequest{TenantId: "other", Name: "docs"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("tenant admin cross-tenant code = %s, want PermissionDenied", status.Code(err))
	}
	if _, err := server.GetCollection(tenantAdmin, &deepdatav3.GetCollectionRequest{TenantId: "acme", Name: "secret"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("tenant admin collection-scope code = %s, want PermissionDenied", status.Code(err))
	}
	if _, err := server.ListCollections(tenantAdmin, &deepdatav3.ListCollectionsRequest{TenantId: "acme"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("collection-scoped tenant admin list code = %s, want PermissionDenied", status.Code(err))
	}
}

func TestCanonicalGRPCTenantLifecycle(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close durable store: %v", err)
		}
	})

	server := &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}
	admin := canonicalGRPCServerAdminContext()

	if _, err := server.CreateTenant(admin, &deepdatav3.CreateTenantRequest{TenantId: "acme"}); err != nil {
		t.Fatalf("create tenant: %v", err)
	}
	if _, err := server.CreateCollection(admin, canonicalGRPCCreateCollectionRequest("acme", "docs")); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	listed, err := server.ListTenants(admin, &deepdatav3.ListTenantsRequest{})
	if err != nil {
		t.Fatalf("list tenants: %v", err)
	}
	var found *deepdatav3.TenantInfo
	for _, tenant := range listed.Tenants {
		if tenant.TenantId == "acme" {
			found = tenant
		}
	}
	if found == nil || found.Status != vcollection.TenantStatusActive {
		t.Fatalf("listed tenant acme = %+v, want status active", found)
	}

	if _, err := server.UpdateTenant(admin, &deepdatav3.UpdateTenantRequest{TenantId: "acme", Status: vcollection.TenantStatusSuspended}); err != nil {
		t.Fatalf("suspend tenant: %v", err)
	}

	insertReq := &deepdatav3.InsertRequest{
		TenantId:   "acme",
		Collection: "docs",
		Vectors:    map[string]*deepdatav3.VectorData{"embedding": denseProtoVector(1, 2)},
	}
	if _, err := server.Insert(admin, insertReq); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("insert on suspended tenant code = %s, want PermissionDenied (err=%v)", status.Code(err), err)
	}

	if _, err := server.UpdateTenant(admin, &deepdatav3.UpdateTenantRequest{TenantId: "acme", Status: vcollection.TenantStatusActive}); err != nil {
		t.Fatalf("reactivate tenant: %v", err)
	}
	if _, err := server.Insert(admin, insertReq); err != nil {
		t.Fatalf("insert on reactivated tenant: %v", err)
	}

	info, err := server.GetTenantInfo(admin, &deepdatav3.GetTenantInfoRequest{TenantId: "acme"})
	if err != nil {
		t.Fatalf("get tenant info: %v", err)
	}
	if info.Tenant == nil || info.Tenant.Usage == nil || info.Tenant.Usage.Documents != 1 {
		t.Fatalf("tenant info usage = %+v, want 1 document", info.Tenant)
	}

	if _, err := server.DeleteTenant(admin, &deepdatav3.DeleteTenantRequest{TenantId: "acme"}); err != nil {
		t.Fatalf("delete tenant: %v", err)
	}
	if _, err := server.DeleteTenant(admin, &deepdatav3.DeleteTenantRequest{TenantId: "acme"}); status.Code(err) != codes.NotFound {
		t.Fatalf("delete missing tenant code = %s, want NotFound (err=%v)", status.Code(err), err)
	}

	if _, err := server.CreateTenant(admin, &deepdatav3.CreateTenantRequest{TenantId: "acme"}); err != nil {
		t.Fatalf("recreate tenant: %v", err)
	}
	if _, err := server.CreateTenant(admin, &deepdatav3.CreateTenantRequest{TenantId: "acme"}); status.Code(err) != codes.AlreadyExists {
		t.Fatalf("duplicate create tenant code = %s, want AlreadyExists (err=%v)", status.Code(err), err)
	}

	tenantAdminNoServerAdmin := canonicalGRPCAdminContext("acme")
	if _, err := server.CreateTenant(tenantAdminNoServerAdmin, &deepdatav3.CreateTenantRequest{TenantId: "other"}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("tenant admin create tenant code = %s, want PermissionDenied (err=%v)", status.Code(err), err)
	}
	if _, err := server.ListTenants(tenantAdminNoServerAdmin, &deepdatav3.ListTenantsRequest{}); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("tenant admin list tenants code = %s, want PermissionDenied (err=%v)", status.Code(err), err)
	}
}

func TestCanonicalGRPCProtoUsesTypedVectorsStructsAndNoIgnoredText(t *testing.T) {
	insert := (&deepdatav3.InsertRequest{}).ProtoReflect().Descriptor()
	if insert.Fields().ByName("text") != nil {
		t.Fatal("ignored insert text field remains in canonical proto")
	}
	assertStructField(t, insert, "metadata")
	assertStructField(t, (&deepdatav3.VectorFieldConfig{}).ProtoReflect().Descriptor(), "index_params")
	assertStructField(t, (&deepdatav3.SearchRequest{}).ProtoReflect().Descriptor(), "filters")
	assertStructField(t, (&deepdatav3.SearchHit{}).ProtoReflect().Descriptor(), "metadata")

	vectors := (&deepdatav3.SearchHit{}).ProtoReflect().Descriptor().Fields().ByName("vectors")
	if vectors == nil || !vectors.IsMap() || vectors.MapValue().Message().FullName() != "deepdata.v3.VectorData" {
		t.Fatalf("SearchHit.vectors is not map<string, VectorData>: %v", vectors)
	}
}

func assertStructField(t *testing.T, descriptor protoreflect.MessageDescriptor, name protoreflect.Name) {
	t.Helper()
	field := descriptor.Fields().ByName(name)
	if field == nil || field.Message() == nil || field.Message().FullName() != "google.protobuf.Struct" {
		t.Fatalf("%s.%s is not google.protobuf.Struct", descriptor.FullName(), name)
	}
}
