package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/sparse"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

// A canonical journal payload is capped at 16 MiB. Protobuf strings can grow
// sixfold when JSON-escaped for the journal, so admit at most 2 MiB of encoded
// mutation input and leave room for the durable envelope.
const canonicalGRPCMutationMaxProtoBytes = 2 << 20

// Search requests never need the historical 64 MiB transport allowance. Keep
// admission bounded before protobuf decoding allocates attacker-controlled
// payloads; mutation handlers apply the stricter journal-safe limit above.
const canonicalGRPCMaxReceiveBytes = 4 << 20

// CollectionGRPCServer implements the canonical tenant-aware DeepData service.
// Every RPC is gated by the same persistence health callback used by the HTTP
// surface so a journal/apply fault cannot leave a read-only gRPC bypass alive.
type CollectionGRPCServer struct {
	deepdatav3.UnimplementedDeepDataServer
	tenants           *vcollection.TenantManager
	persistenceHealth func() error
	embedder          *serverEmbedder // process text embedder for `texts`; nil = none
}

func (s *CollectionGRPCServer) GetTenantInfo(ctx context.Context, req *deepdatav3.GetTenantInfoRequest) (*deepdatav3.GetTenantInfoResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, "request required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, "", "admin"); err != nil {
		return nil, err
	}

	listed, err := s.tenants.ListCollectionInfosChecked(req.TenantId)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	infos := append([]vcollection.CollectionInfo(nil), listed...)
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })
	stats := make([]*deepdatav3.CollectionStats, len(infos))
	var totalDocuments uint64
	for i, info := range infos {
		docCount := nonNegativeUint64(info.DocCount)
		totalDocuments += docCount
		stats[i] = &deepdatav3.CollectionStats{
			Name:          info.Name,
			DocumentCount: docCount,
			FieldCount:    int32(len(info.Fields)),
		}
	}
	resp := &deepdatav3.GetTenantInfoResponse{
		TenantId:        req.TenantId,
		CollectionCount: uint64(len(infos)),
		TotalDocuments:  totalDocuments,
		Collections:     stats,
	}
	if info, err := s.tenants.GetTenantInfo(req.TenantId); err == nil {
		resp.Tenant = tenantInfoProto(info)
	}
	return resp, nil
}

// CreateTenant provisions a new tenant record. Tenant lifecycle crosses
// tenant boundaries by nature, so it is server-administrator only.
func (s *CollectionGRPCServer) CreateTenant(ctx context.Context, req *deepdatav3.CreateTenantRequest) (*deepdatav3.CreateTenantResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || !isValidTenantID(req.TenantId) {
		return nil, apierror.New(apierror.CodeInvalidArgument, "valid tenant_id required").GRPC(ctx)
	}
	if _, err := authorizeServerAdminGRPC(ctx); err != nil {
		return nil, err
	}
	if err := s.tenants.CreateTenant(ctx, tenantRecordFromProto(req.TenantId, req.Status, req.Quota)); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInvalidArgument)
	}
	return &deepdatav3.CreateTenantResponse{TenantId: req.TenantId}, nil
}

// ListTenants returns every tenant record. Server-administrator only.
func (s *CollectionGRPCServer) ListTenants(ctx context.Context, req *deepdatav3.ListTenantsRequest) (*deepdatav3.ListTenantsResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if _, err := authorizeServerAdminGRPC(ctx); err != nil {
		return nil, err
	}
	infos, err := s.tenants.ListTenantInfos()
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	tenants := make([]*deepdatav3.TenantInfo, len(infos))
	for i := range infos {
		tenants[i] = tenantInfoProto(infos[i])
	}
	return &deepdatav3.ListTenantsResponse{Tenants: tenants}, nil
}

// UpdateTenant upserts a tenant record; PUT on an unknown tenant creates it,
// so this also covers suspend/reactivate transitions. Server-administrator
// only.
func (s *CollectionGRPCServer) UpdateTenant(ctx context.Context, req *deepdatav3.UpdateTenantRequest) (*deepdatav3.UpdateTenantResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || !isValidTenantID(req.TenantId) {
		return nil, apierror.New(apierror.CodeInvalidArgument, "valid tenant_id required").GRPC(ctx)
	}
	if _, err := authorizeServerAdminGRPC(ctx); err != nil {
		return nil, err
	}
	if err := s.tenants.UpdateTenant(ctx, tenantRecordFromProto(req.TenantId, req.Status, req.Quota)); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInvalidArgument)
	}
	return &deepdatav3.UpdateTenantResponse{TenantId: req.TenantId}, nil
}

// DeleteTenant removes a tenant's record and every collection it owns.
// Server-administrator only.
func (s *CollectionGRPCServer) DeleteTenant(ctx context.Context, req *deepdatav3.DeleteTenantRequest) (*deepdatav3.DeleteTenantResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || !isValidTenantID(req.TenantId) {
		return nil, apierror.New(apierror.CodeInvalidArgument, "valid tenant_id required").GRPC(ctx)
	}
	if _, err := authorizeServerAdminGRPC(ctx); err != nil {
		return nil, err
	}
	if err := s.tenants.DeleteTenant(ctx, req.TenantId); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	return &deepdatav3.DeleteTenantResponse{TenantId: req.TenantId}, nil
}

func (s *CollectionGRPCServer) ListCollections(ctx context.Context, req *deepdatav3.ListCollectionsRequest) (*deepdatav3.ListCollectionsResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, "request required").GRPC(ctx)
	}
	// Discovery is read-gated: listing collection names and schemas is the
	// same class of read as searching them (CTL-04).
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, "", "read"); err != nil {
		return nil, err
	}

	listed, err := s.tenants.ListCollectionInfosChecked(req.TenantId)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	infos := append([]vcollection.CollectionInfo(nil), listed...)
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })
	collections := make([]*deepdatav3.CollectionInfo, len(infos))
	for i := range infos {
		converted, err := collectionInfoToProto(infos[i])
		if err != nil {
			return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode collection %q: %v", infos[i].Name, err)).GRPC(ctx)
		}
		collections[i] = converted
	}
	return &deepdatav3.ListCollectionsResponse{Collections: collections}, nil
}

func (s *CollectionGRPCServer) GetCollection(ctx context.Context, req *deepdatav3.GetCollectionRequest) (*deepdatav3.GetCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Name == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection name required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "read"); err != nil {
		return nil, err
	}
	info, err := s.tenants.GetCollectionInfo(req.TenantId, req.Name)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
	}
	converted, err := collectionInfoToProto(*info)
	if err != nil {
		return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode collection %q: %v", req.Name, err)).GRPC(ctx)
	}
	return &deepdatav3.GetCollectionResponse{Collection: converted}, nil
}

func (s *CollectionGRPCServer) CreateCollection(ctx context.Context, req *deepdatav3.CreateCollectionRequest) (*deepdatav3.CreateCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, "request required").GRPC(ctx)
	}
	if !vcollection.IsValidCanonicalIdentifier(req.Name) {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection name must be 1-64 alphanumeric/hyphen/underscore characters").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "admin"); err != nil {
		return nil, err
	}
	if req.Name == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection name required").GRPC(ctx)
	}
	if len(req.Fields) == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "at least one vector field required").GRPC(ctx)
	}

	fields := make([]vcollection.VectorField, len(req.Fields))
	for i, field := range req.Fields {
		if field == nil {
			return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("field %d is required", i)).GRPC(ctx)
		}
		idxType, err := parseIndexType(field.IndexType)
		if err != nil {
			return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("field %d index_type: %v", i, err)).GRPC(ctx)
		}
		fields[i] = vcollection.VectorField{
			Name: field.Name,
			Type: vcollection.VectorType(field.Type),
			Dim:  int(field.Dim),
			Index: vcollection.IndexConfig{
				Type:   idxType,
				Params: structToMap(field.IndexParams),
			},
		}
		if field.Embedding != nil {
			fields[i].Embedding = &vcollection.EmbeddingConfig{Provider: field.Embedding.Provider, Model: field.Embedding.Model}
		}
	}

	schema := vcollection.CollectionSchema{
		Name:        req.Name,
		Fields:      fields,
		Metadata:    structToMap(req.Metadata),
		Description: req.Description,
		Durability:  req.Durability,
	}
	if aerr := resolveSchemaEmbedding(&schema, s.embedder); aerr != nil {
		return nil, aerr.GRPC(ctx)
	}
	if _, err := s.tenants.CreateCollection(ctx, req.TenantId, schema); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInvalidArgument)
	}
	return &deepdatav3.CreateCollectionResponse{Name: req.Name}, nil
}

func (s *CollectionGRPCServer) DeleteCollection(ctx context.Context, req *deepdatav3.DeleteCollectionRequest) (*deepdatav3.DeleteCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Name == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection name required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "admin"); err != nil {
		return nil, err
	}
	if err := s.tenants.DeleteCollection(ctx, req.TenantId, req.Name); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
	}
	return &deepdatav3.DeleteCollectionResponse{}, nil
}

func (s *CollectionGRPCServer) Insert(ctx context.Context, req *deepdatav3.InsertRequest) (*deepdatav3.InsertResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := requireCanonicalGRPCMutationSize(ctx, req); err != nil {
		return nil, err
	}
	if len(req.Vectors) == 0 && len(req.Texts) == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "at least one vector or text required").GRPC(ctx)
	}

	vectors, err := protoVectorsToInterface(req.Vectors)
	if err != nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("%v", err)).GRPC(ctx)
	}
	if _, err := s.applyTexts(ctx, req.TenantId, req.Collection, req.Texts, vectors, false); err != nil {
		return nil, err
	}
	doc := &vcollection.Document{ID: req.Id, Vectors: toDocumentVectors(vectors), Metadata: structToMap(req.Metadata)}
	if err := s.tenants.AddDocument(ctx, req.TenantId, req.Collection, doc); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	return &deepdatav3.InsertResponse{Id: doc.ID}, nil
}

func (s *CollectionGRPCServer) BatchInsert(ctx context.Context, req *deepdatav3.BatchInsertRequest) (*deepdatav3.BatchInsertResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if len(req.Docs) == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "at least one document required").GRPC(ctx)
	}
	if len(req.Docs) > vcollection.MaxBatchDocuments {
		return nil, apierror.New(apierror.CodePayloadTooLarge, fmt.Sprintf("batch too large: maximum is %d documents", vcollection.MaxBatchDocuments)).GRPC(ctx)
	}
	if err := requireCanonicalGRPCMutationSize(ctx, req); err != nil {
		return nil, err
	}

	docs := make([]vcollection.Document, len(req.Docs))
	var fields []vcollection.FieldInfo // schema, loaded once if any document sends texts
	for i, batchDoc := range req.Docs {
		if batchDoc == nil {
			return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("document %d is required", i)).GRPC(ctx)
		}
		if len(batchDoc.Vectors) == 0 && len(batchDoc.Texts) == 0 {
			return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("document %d requires at least one vector or text", i)).GRPC(ctx)
		}
		vectors, err := protoVectorsToInterface(batchDoc.Vectors)
		if err != nil {
			return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("document %d: %v", i, err)).GRPC(ctx)
		}
		if len(batchDoc.Texts) > 0 {
			if fields == nil {
				info, err := s.tenants.GetCollectionInfo(req.TenantId, req.Collection)
				if err != nil {
					return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
				}
				fields = info.Fields
			}
			if _, aerr := resolveTexts(fields, s.embedder, batchDoc.Texts, vectors, false); aerr != nil {
				aerr.Message = fmt.Sprintf("document %d: %s", i, aerr.Message)
				return nil, aerr.GRPC(ctx)
			}
		}
		docs[i] = vcollection.Document{ID: batchDoc.Id, Vectors: toDocumentVectors(vectors), Metadata: structToMap(batchDoc.Metadata)}
	}
	if err := s.tenants.BatchAddDocuments(ctx, req.TenantId, req.Collection, docs); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}

	ids := make([]uint64, len(docs))
	for i := range docs {
		ids[i] = docs[i].ID
	}
	return &deepdatav3.BatchInsertResponse{Ids: ids, Inserted: int32(len(docs))}, nil
}

func (s *CollectionGRPCServer) Search(ctx context.Context, req *deepdatav3.SearchRequest) (*deepdatav3.SearchResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "read"); err != nil {
		return nil, err
	}
	queries, err := protoVectorsToInterface(req.Queries)
	if err != nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("%v", err)).GRPC(ctx)
	}
	embeddedBy, err := s.applyTexts(ctx, req.TenantId, req.Collection, req.Texts, queries, true)
	if err != nil {
		return nil, err
	}
	searchReq := vcollection.SearchRequest{
		CollectionName: req.Collection,
		Queries:        queries,
		TopK:           int(req.TopK),
		EfSearch:       int(req.EfSearch),
		Filters:        structToMap(req.Filters),
	}
	if req.IncludeVectors {
		include := true
		searchReq.IncludeVectors = &include
	}
	if req.HybridParams != nil {
		searchReq.HybridParams = &vcollection.HybridSearchParams{
			Strategy:    req.HybridParams.Strategy,
			Weights:     cloneStringFloat32Map(req.HybridParams.Weights),
			RRFConstant: req.HybridParams.RrfConstant,
		}
	}
	searchReq.ScoreFloor = req.ScoreFloor
	searchReq.UsageBoost = req.UsageBoost
	if req.Fallback != nil {
		searchReq.Fallback = &vcollection.FallbackParams{
			Primary:   req.Fallback.Primary,
			Secondary: req.Fallback.Secondary,
			Threshold: req.Fallback.Threshold,
		}
	}

	resp, err := s.tenants.SearchCollection(ctx, req.TenantId, searchReq)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	hits := make([]*deepdatav3.SearchHit, len(resp.Documents))
	for i, doc := range resp.Documents {
		metadata, err := mapToStruct(doc.Metadata)
		if err != nil {
			return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode result %d metadata: %v", i, err)).GRPC(ctx)
		}
		vectors, err := interfaceVectorsToProto(doc.Vectors)
		if err != nil {
			return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode result %d vectors: %v", i, err)).GRPC(ctx)
		}
		hit := &deepdatav3.SearchHit{Id: doc.ID, Metadata: metadata, Vectors: vectors}
		if i < len(resp.Scores) {
			hit.Score = resp.Scores[i]
		}
		hits[i] = hit
	}
	return &deepdatav3.SearchResponse{
		Results:            hits,
		CandidatesExamined: int32(resp.CandidatesExamined),
		BestScore:          resp.BestScore,
		WeakMatch:          resp.WeakMatch,
		FellBackTo:         resp.FellBackTo,
		EmbeddedBy:         embeddedBy,
		ScoreDirection:     resp.ScoreDirection,
		QueryTimeMs:        resp.QueryTimeMs,
		RequestId:          requestIDFromContext(ctx),
	}, nil
}

func (s *CollectionGRPCServer) DeleteDoc(ctx context.Context, req *deepdatav3.DeleteDocRequest) (*deepdatav3.DeleteDocResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" || req.DocId == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection and non-zero doc_id required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := s.tenants.DeleteDocument(ctx, req.TenantId, req.Collection, req.DocId); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
	}
	return &deepdatav3.DeleteDocResponse{}, nil
}

func (s *CollectionGRPCServer) Upsert(ctx context.Context, req *deepdatav3.UpsertRequest) (*deepdatav3.UpsertResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection required").GRPC(ctx)
	}
	if req.Id == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "upsert requires a caller-supplied non-zero id").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := requireCanonicalGRPCMutationSize(ctx, req); err != nil {
		return nil, err
	}
	if len(req.Vectors) == 0 && len(req.Texts) == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "at least one vector or text required").GRPC(ctx)
	}

	vectors, err := protoVectorsToInterface(req.Vectors)
	if err != nil {
		return nil, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("%v", err)).GRPC(ctx)
	}
	if _, err := s.applyTexts(ctx, req.TenantId, req.Collection, req.Texts, vectors, false); err != nil {
		return nil, err
	}
	doc := &vcollection.Document{ID: req.Id, Vectors: toDocumentVectors(vectors), Metadata: structToMap(req.Metadata)}
	if err := s.tenants.UpsertDocument(ctx, req.TenantId, req.Collection, doc); err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeInternal)
	}
	return &deepdatav3.UpsertResponse{Id: req.Id}, nil
}

func (s *CollectionGRPCServer) GetDoc(ctx context.Context, req *deepdatav3.GetDocRequest) (*deepdatav3.GetDocResponse, error) {
	if err := s.requirePersistenceHealthy(ctx); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" || req.DocId == 0 {
		return nil, apierror.New(apierror.CodeInvalidArgument, "collection and non-zero doc_id required").GRPC(ctx)
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "read"); err != nil {
		return nil, err
	}
	doc, err := s.tenants.GetDocumentChecked(req.TenantId, req.Collection, req.DocId)
	if err != nil {
		return nil, canonicalGRPCError(ctx, err, apierror.CodeNotFound)
	}
	vectors, err := interfaceVectorsToProto(doc.Vectors)
	if err != nil {
		return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode document vectors: %v", err)).GRPC(ctx)
	}
	metadata, err := mapToStruct(doc.Metadata)
	if err != nil {
		return nil, apierror.New(apierror.CodeInternal, fmt.Sprintf("encode document metadata: %v", err)).GRPC(ctx)
	}
	return &deepdatav3.GetDocResponse{Id: doc.ID, Vectors: vectors, Metadata: metadata}, nil
}

func (s *CollectionGRPCServer) requirePersistenceHealthy(ctx context.Context) error {
	if s == nil || s.tenants == nil || s.persistenceHealth == nil {
		return apierror.New(apierror.CodeUnavailable, "collection persistence unavailable").GRPC(ctx)
	}
	if err := s.persistenceHealth(); err != nil {
		return apierror.New(apierror.CodeUnavailable, "collection persistence unavailable").GRPC(ctx)
	}
	return nil
}

func requireCanonicalGRPCMutationSize(ctx context.Context, message proto.Message) error {
	if message == nil {
		return apierror.New(apierror.CodeInvalidArgument, "request required").GRPC(ctx)
	}
	if size := proto.Size(message); size > canonicalGRPCMutationMaxProtoBytes {
		return apierror.New(apierror.CodePayloadTooLarge, fmt.Sprintf("mutation request is %d bytes; maximum is %d", size, canonicalGRPCMutationMaxProtoBytes)).GRPC(ctx)
	}
	return nil
}

// canonicalGRPCError projects an engine error onto the wire. A cancelled or
// expired caller context stays a plain status; everything else goes through
// apierror.FromEngine, with fallback as the code for the unclassified rest.
func canonicalGRPCError(ctx context.Context, err error, fallback string) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, err.Error())
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, err.Error())
	}
	return apierror.FromEngine(err, fallback).GRPC(ctx)
}

func protoVectorsToInterface(vectors map[string]*deepdatav3.VectorData) (map[string]interface{}, error) {
	converted := make(map[string]interface{}, len(vectors))
	for name, vector := range vectors {
		if name == "" {
			return nil, errors.New("vector field name cannot be empty")
		}
		value, err := vectorDataToInterface(vector)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", name, err)
		}
		converted[name] = value
	}
	return converted, nil
}

func vectorDataToInterface(vector *deepdatav3.VectorData) (vcollection.Vector, error) {
	if vector == nil {
		return vcollection.Vector{}, errors.New("nil vector data")
	}
	switch data := vector.Data.(type) {
	case *deepdatav3.VectorData_Dense:
		if data.Dense == nil || len(data.Dense.Values) == 0 {
			return vcollection.Vector{}, errors.New("dense vector cannot be empty")
		}
		if err := validateFiniteFloat32(data.Dense.Values); err != nil {
			return vcollection.Vector{}, fmt.Errorf("dense vector: %w", err)
		}
		return vcollection.Vector{Dense: data.Dense.Values}, nil
	case *deepdatav3.VectorData_Sparse:
		if data.Sparse == nil || data.Sparse.Dim <= 0 {
			return vcollection.Vector{}, errors.New("sparse vector dimension must be positive")
		}
		if err := validateFiniteFloat32(data.Sparse.Values); err != nil {
			return vcollection.Vector{}, fmt.Errorf("sparse vector: %w", err)
		}
		sv, err := sparse.NewSparseVector(data.Sparse.Indices, data.Sparse.Values, int(data.Sparse.Dim))
		if err != nil {
			return vcollection.Vector{}, err
		}
		return vcollection.Vector{Sparse: sv}, nil
	default:
		return vcollection.Vector{}, errors.New("vector data must contain dense or sparse values")
	}
}

func interfaceVectorsToProto(vectors map[string]vcollection.Vector) (map[string]*deepdatav3.VectorData, error) {
	if len(vectors) == 0 {
		return nil, nil
	}
	converted := make(map[string]*deepdatav3.VectorData, len(vectors))
	for name, value := range vectors {
		vector, err := vectorInterfaceToProto(value)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", name, err)
		}
		converted[name] = vector
	}
	return converted, nil
}

func vectorInterfaceToProto(value vcollection.Vector) (*deepdatav3.VectorData, error) {
	switch {
	case value.Dense != nil:
		values := append([]float32(nil), value.Dense...)
		if err := validateFiniteFloat32(values); err != nil {
			return nil, err
		}
		return denseVectorData(values), nil
	case value.Sparse != nil:
		return sparseVectorData(value.Sparse)
	default:
		return nil, fmt.Errorf("unsupported vector representation %T", value)
	}
}

func denseVectorData(values []float32) *deepdatav3.VectorData {
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Dense{Dense: &deepdatav3.DenseVector{Values: values}}}
}

func sparseVectorData(vector *sparse.SparseVector) (*deepdatav3.VectorData, error) {
	if vector.Dim <= 0 || vector.Dim > math.MaxInt32 {
		return nil, fmt.Errorf("sparse dimension %d is outside int32 range", vector.Dim)
	}
	if err := validateFiniteFloat32(vector.Values); err != nil {
		return nil, err
	}
	validated, err := sparse.NewSparseVector(vector.Indices, vector.Values, vector.Dim)
	if err != nil {
		return nil, err
	}
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Sparse{Sparse: &deepdatav3.SparseVector{
		Indices: append([]uint32(nil), validated.Indices...),
		Values:  append([]float32(nil), validated.Values...),
		Dim:     int32(validated.Dim),
	}}}, nil
}

func collectionInfoToProto(info vcollection.CollectionInfo) (*deepdatav3.CollectionInfo, error) {
	fields := make([]*deepdatav3.VectorFieldConfig, len(info.Fields))
	for i, field := range info.Fields {
		if field.Dim < math.MinInt32 || field.Dim > math.MaxInt32 {
			return nil, fmt.Errorf("field %q dimension is outside int32 range", field.Name)
		}
		params, err := mapToStruct(field.Index.Params)
		if err != nil {
			return nil, fmt.Errorf("field %q index parameters: %w", field.Name, err)
		}
		fields[i] = &deepdatav3.VectorFieldConfig{
			Name:        field.Name,
			Type:        int32(field.Type),
			Dim:         int32(field.Dim),
			IndexType:   field.Index.Type.String(),
			IndexParams: params,

			ScoreDirection: field.ScoreDirection,
		}
		if field.Embedding != nil {
			fields[i].Embedding = &deepdatav3.EmbeddingConfig{Provider: field.Embedding.Provider, Model: field.Embedding.Model}
		}
	}
	metadata, err := mapToStruct(info.Metadata)
	if err != nil {
		return nil, err
	}
	return &deepdatav3.CollectionInfo{
		Name:          info.Name,
		Fields:        fields,
		Description:   info.Description,
		Metadata:      metadata,
		DocumentCount: nonNegativeUint64(info.DocCount),
		Durability:    info.Durability,
	}, nil
}

func mapToStruct(values map[string]interface{}) (*structpb.Struct, error) {
	if len(values) == 0 {
		return nil, nil
	}
	return structpb.NewStruct(values)
}

func structToMap(value *structpb.Struct) map[string]interface{} {
	if value == nil {
		return nil
	}
	return value.AsMap()
}

func cloneStringFloat32Map(values map[string]float32) map[string]float32 {
	if len(values) == 0 {
		return nil
	}
	clone := make(map[string]float32, len(values))
	for key, value := range values {
		clone[key] = value
	}
	return clone
}

func validateFiniteFloat32(values []float32) error {
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("element %d must be finite", i)
		}
	}
	return nil
}

func nonNegativeUint64(value int) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value)
}

func parseIndexType(value string) (vcollection.IndexType, error) {
	if value == "" {
		return vcollection.IndexTypeHNSW, nil
	}
	return vcollection.ParseIndexType(value)
}

func authorizeCanonicalGRPC(ctx context.Context, tenantID, collection, permission string) error {
	if !isValidTenantID(tenantID) {
		return apierror.New(apierror.CodeInvalidArgument, "valid tenant_id required").GRPC(ctx)
	}
	tenantCtx, _ := security.GetTenantContextFromContext(ctx)
	if err := security.AuthorizeTenantAccess(tenantCtx, tenantID, collection, permission); err != nil {
		return canonicalGRPCAuthError(ctx, err)
	}
	return nil
}

// authorizeServerAdminGRPC requires the global server-administrator
// credential; tenant lifecycle RPCs cross tenant boundaries by nature, so no
// tenant or collection scope applies.
func authorizeServerAdminGRPC(ctx context.Context) (*security.TenantContext, error) {
	tenantCtx, _ := security.GetTenantContextFromContext(ctx)
	if err := security.AuthorizeServerAdmin(tenantCtx); err != nil {
		return nil, canonicalGRPCAuthError(ctx, err)
	}
	return tenantCtx, nil
}

// canonicalGRPCAuthError projects a security.AuthorizationError onto the
// wire: unauthenticated stays unauthenticated, everything else is a
// permission denial.
func canonicalGRPCAuthError(ctx context.Context, err error) error {
	if security.IsAuthorizationFailure(err, security.AuthorizationUnauthenticated) {
		return apierror.New(apierror.CodeUnauthenticated, err.Error()).GRPC(ctx)
	}
	return apierror.New(apierror.CodePermissionDenied, err.Error()).GRPC(ctx)
}

// tenantInfoProto converts a tenant's lifecycle snapshot to the wire type.
func tenantInfoProto(info vcollection.TenantInfo) *deepdatav3.TenantInfo {
	return &deepdatav3.TenantInfo{
		TenantId: info.TenantID,
		Status:   info.Status,
		Quota: &deepdatav3.TenantQuota{
			MaxDocuments:   info.Quota.MaxDocuments,
			MaxBytes:       info.Quota.MaxBytes,
			MaxCollections: info.Quota.MaxCollections,
		},
		Usage: &deepdatav3.TenantUsage{
			Documents:   info.Usage.Documents,
			Bytes:       info.Usage.Bytes,
			Collections: info.Usage.Collections,
		},
	}
}

// tenantRecordFromProto decodes a tenant lifecycle request into the domain
// type. An empty status defaults to active: the common case is provisioning
// a tenant ready for immediate use, not a pre-suspended one.
func tenantRecordFromProto(tenantID, status string, quota *deepdatav3.TenantQuota) vcollection.TenantRecord {
	if status == "" {
		status = vcollection.TenantStatusActive
	}
	rec := vcollection.TenantRecord{TenantID: tenantID, Status: status}
	if quota != nil {
		rec.Quota = vcollection.TenantQuota{
			MaxDocuments:   quota.MaxDocuments,
			MaxBytes:       quota.MaxBytes,
			MaxCollections: quota.MaxCollections,
		}
	}
	return rec
}
