package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
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
}

func (s *CollectionGRPCServer) GetTenantInfo(ctx context.Context, req *deepdatav3.GetTenantInfoRequest) (*deepdatav3.GetTenantInfoResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, "", "admin"); err != nil {
		return nil, err
	}

	listed, err := s.tenants.ListCollectionInfosChecked(req.TenantId)
	if err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
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
	return &deepdatav3.GetTenantInfoResponse{
		TenantId:        req.TenantId,
		CollectionCount: uint64(len(infos)),
		TotalDocuments:  totalDocuments,
		Collections:     stats,
	}, nil
}

func (s *CollectionGRPCServer) ListCollections(ctx context.Context, req *deepdatav3.ListCollectionsRequest) (*deepdatav3.ListCollectionsResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, "", "admin"); err != nil {
		return nil, err
	}

	listed, err := s.tenants.ListCollectionInfosChecked(req.TenantId)
	if err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
	}
	infos := append([]vcollection.CollectionInfo(nil), listed...)
	sort.Slice(infos, func(i, j int) bool { return infos[i].Name < infos[j].Name })
	collections := make([]*deepdatav3.CollectionInfo, len(infos))
	for i := range infos {
		converted, err := collectionInfoToProto(infos[i])
		if err != nil {
			return nil, status.Errorf(codes.Internal, "encode collection %q: %v", infos[i].Name, err)
		}
		collections[i] = converted
	}
	return &deepdatav3.ListCollectionsResponse{Collections: collections}, nil
}

func (s *CollectionGRPCServer) GetCollection(ctx context.Context, req *deepdatav3.GetCollectionRequest) (*deepdatav3.GetCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "collection name required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "read"); err != nil {
		return nil, err
	}
	info, err := s.tenants.GetCollectionInfo(req.TenantId, req.Name)
	if err != nil {
		return nil, canonicalGRPCError(err, codes.NotFound)
	}
	converted, err := collectionInfoToProto(*info)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "encode collection %q: %v", req.Name, err)
	}
	return &deepdatav3.GetCollectionResponse{Collection: converted}, nil
}

func (s *CollectionGRPCServer) CreateCollection(ctx context.Context, req *deepdatav3.CreateCollectionRequest) (*deepdatav3.CreateCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "request required")
	}
	if !vcollection.IsValidCanonicalIdentifier(req.Name) {
		return nil, status.Error(codes.InvalidArgument, "collection name must be 1-64 alphanumeric/hyphen/underscore characters")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "admin"); err != nil {
		return nil, err
	}
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "collection name required")
	}
	if len(req.Fields) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one vector field required")
	}

	fields := make([]vcollection.VectorField, len(req.Fields))
	for i, field := range req.Fields {
		if field == nil {
			return nil, status.Errorf(codes.InvalidArgument, "field %d is required", i)
		}
		idxType, err := parseIndexType(field.IndexType)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "field %d index_type: %v", i, err)
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
	}

	schema := vcollection.CollectionSchema{
		Name:        req.Name,
		Fields:      fields,
		Metadata:    structToMap(req.Metadata),
		Description: req.Description,
	}
	if _, err := s.tenants.CreateCollection(ctx, req.TenantId, schema); err != nil {
		if strings.Contains(err.Error(), "already exists") {
			return nil, canonicalGRPCError(err, codes.AlreadyExists)
		}
		return nil, canonicalGRPCError(err, codes.InvalidArgument)
	}
	return &deepdatav3.CreateCollectionResponse{Name: req.Name}, nil
}

func (s *CollectionGRPCServer) DeleteCollection(ctx context.Context, req *deepdatav3.DeleteCollectionRequest) (*deepdatav3.DeleteCollectionResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "collection name required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Name, "admin"); err != nil {
		return nil, err
	}
	if err := s.tenants.DeleteCollection(ctx, req.TenantId, req.Name); err != nil {
		return nil, canonicalGRPCError(err, codes.NotFound)
	}
	return &deepdatav3.DeleteCollectionResponse{}, nil
}

func (s *CollectionGRPCServer) Insert(ctx context.Context, req *deepdatav3.InsertRequest) (*deepdatav3.InsertResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := requireCanonicalGRPCMutationSize(req); err != nil {
		return nil, err
	}
	if len(req.Vectors) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one vector required")
	}

	vectors, err := protoVectorsToInterface(req.Vectors)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}
	doc := &vcollection.Document{ID: req.Id, Vectors: vectors, Metadata: structToMap(req.Metadata)}
	if err := s.tenants.AddDocument(ctx, req.TenantId, req.Collection, doc); err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
	}
	return &deepdatav3.InsertResponse{Id: doc.ID}, nil
}

func (s *CollectionGRPCServer) BatchInsert(ctx context.Context, req *deepdatav3.BatchInsertRequest) (*deepdatav3.BatchInsertResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if len(req.Docs) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one document required")
	}
	if len(req.Docs) > vcollection.CanonicalMaxBatchDocuments {
		return nil, status.Errorf(codes.ResourceExhausted, "batch too large: maximum is %d documents", vcollection.CanonicalMaxBatchDocuments)
	}
	if err := requireCanonicalGRPCMutationSize(req); err != nil {
		return nil, err
	}

	docs := make([]vcollection.Document, len(req.Docs))
	for i, batchDoc := range req.Docs {
		if batchDoc == nil {
			return nil, status.Errorf(codes.InvalidArgument, "document %d is required", i)
		}
		if len(batchDoc.Vectors) == 0 {
			return nil, status.Errorf(codes.InvalidArgument, "document %d requires at least one vector", i)
		}
		vectors, err := protoVectorsToInterface(batchDoc.Vectors)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "document %d: %v", i, err)
		}
		docs[i] = vcollection.Document{ID: batchDoc.Id, Vectors: vectors, Metadata: structToMap(batchDoc.Metadata)}
	}
	if err := s.tenants.BatchAddDocuments(ctx, req.TenantId, req.Collection, docs); err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
	}

	ids := make([]uint64, len(docs))
	for i := range docs {
		ids[i] = docs[i].ID
	}
	return &deepdatav3.BatchInsertResponse{Ids: ids, Inserted: int32(len(docs))}, nil
}

func (s *CollectionGRPCServer) Search(ctx context.Context, req *deepdatav3.SearchRequest) (*deepdatav3.SearchResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "read"); err != nil {
		return nil, err
	}
	if req.TopK <= 0 {
		return nil, status.Error(codes.InvalidArgument, "top_k must be positive")
	}
	if req.TopK > vcollection.CanonicalMaxSearchTopK {
		return nil, status.Errorf(codes.InvalidArgument, "top_k must not exceed %d", vcollection.CanonicalMaxSearchTopK)
	}
	if req.EfSearch < 0 {
		return nil, status.Error(codes.InvalidArgument, "ef_search cannot be negative")
	}
	if len(req.Queries) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one query vector required")
	}
	if len(req.Queries) > vcollection.CanonicalMaxSearchFields {
		return nil, status.Errorf(codes.InvalidArgument, "at most %d query fields are supported", vcollection.CanonicalMaxSearchFields)
	}
	if len(req.Queries) > 1 && req.HybridParams == nil {
		return nil, status.Error(codes.InvalidArgument, "multiple query fields require hybrid_params")
	}

	queries, err := protoVectorsToInterface(req.Queries)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
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

	resp, err := s.tenants.SearchCollection(ctx, req.TenantId, searchReq)
	if err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
	}
	hits := make([]*deepdatav3.SearchHit, len(resp.Documents))
	for i, doc := range resp.Documents {
		metadata, err := mapToStruct(doc.Metadata)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "encode result %d metadata: %v", i, err)
		}
		vectors, err := interfaceVectorsToProto(doc.Vectors)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "encode result %d vectors: %v", i, err)
		}
		hit := &deepdatav3.SearchHit{Id: doc.ID, Metadata: metadata, Vectors: vectors}
		if i < len(resp.Scores) {
			hit.Score = resp.Scores[i]
		}
		hits[i] = hit
	}
	return &deepdatav3.SearchResponse{Results: hits, CandidatesExamined: int32(resp.CandidatesExamined)}, nil
}

func (s *CollectionGRPCServer) DeleteDoc(ctx context.Context, req *deepdatav3.DeleteDocRequest) (*deepdatav3.DeleteDocResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" || req.DocId == 0 {
		return nil, status.Error(codes.InvalidArgument, "collection and non-zero doc_id required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := s.tenants.DeleteDocument(ctx, req.TenantId, req.Collection, req.DocId); err != nil {
		return nil, canonicalGRPCError(err, codes.NotFound)
	}
	return &deepdatav3.DeleteDocResponse{}, nil
}

func (s *CollectionGRPCServer) Upsert(ctx context.Context, req *deepdatav3.UpsertRequest) (*deepdatav3.UpsertResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if req.Id == 0 {
		return nil, status.Error(codes.InvalidArgument, "upsert requires a caller-supplied non-zero id")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "write"); err != nil {
		return nil, err
	}
	if err := requireCanonicalGRPCMutationSize(req); err != nil {
		return nil, err
	}
	if len(req.Vectors) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one vector required")
	}

	vectors, err := protoVectorsToInterface(req.Vectors)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}
	doc := &vcollection.Document{ID: req.Id, Vectors: vectors, Metadata: structToMap(req.Metadata)}
	if err := s.tenants.UpsertDocument(ctx, req.TenantId, req.Collection, doc); err != nil {
		return nil, canonicalGRPCError(err, codes.Internal)
	}
	return &deepdatav3.UpsertResponse{Id: req.Id}, nil
}

func (s *CollectionGRPCServer) GetDoc(ctx context.Context, req *deepdatav3.GetDocRequest) (*deepdatav3.GetDocResponse, error) {
	if err := s.requirePersistenceHealthy(); err != nil {
		return nil, err
	}
	if req == nil || req.Collection == "" || req.DocId == 0 {
		return nil, status.Error(codes.InvalidArgument, "collection and non-zero doc_id required")
	}
	if err := authorizeCanonicalGRPC(ctx, req.TenantId, req.Collection, "read"); err != nil {
		return nil, err
	}
	doc, ok := s.tenants.GetDocument(req.TenantId, req.Collection, req.DocId)
	if !ok {
		return nil, status.Errorf(codes.NotFound, "document %d not found in collection %s", req.DocId, req.Collection)
	}
	vectors, err := interfaceVectorsToProto(doc.Vectors)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "encode document vectors: %v", err)
	}
	metadata, err := mapToStruct(doc.Metadata)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "encode document metadata: %v", err)
	}
	return &deepdatav3.GetDocResponse{Id: doc.ID, Vectors: vectors, Metadata: metadata}, nil
}

func (s *CollectionGRPCServer) requirePersistenceHealthy() error {
	if s == nil || s.tenants == nil || s.persistenceHealth == nil {
		return status.Error(codes.Unavailable, "collection persistence unavailable")
	}
	if err := s.persistenceHealth(); err != nil {
		return status.Error(codes.Unavailable, "collection persistence unavailable")
	}
	return nil
}

func requireCanonicalGRPCMutationSize(message proto.Message) error {
	if message == nil {
		return status.Error(codes.InvalidArgument, "request required")
	}
	if size := proto.Size(message); size > canonicalGRPCMutationMaxProtoBytes {
		return status.Errorf(codes.ResourceExhausted, "mutation request is %d bytes; maximum is %d", size, canonicalGRPCMutationMaxProtoBytes)
	}
	return nil
}

func canonicalGRPCError(err error, fallback codes.Code) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, err.Error())
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, err.Error())
	case errors.Is(err, vcollection.ErrDurableStoreClosed), errors.Is(err, vcollection.ErrDurableStoreFaulted):
		return status.Error(codes.Unavailable, "collection persistence unavailable")
	case errors.Is(err, vcollection.ErrTenantLimitExceeded),
		errors.Is(err, vcollection.ErrCollectionLimitExceeded),
		errors.Is(err, vcollection.ErrSearchResponseBudgetExceeded):
		return status.Error(codes.ResourceExhausted, err.Error())
	default:
		return status.Error(fallback, err.Error())
	}
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

func vectorDataToInterface(vector *deepdatav3.VectorData) (interface{}, error) {
	if vector == nil {
		return nil, errors.New("nil vector data")
	}
	switch data := vector.Data.(type) {
	case *deepdatav3.VectorData_Dense:
		if data.Dense == nil || len(data.Dense.Values) == 0 {
			return nil, errors.New("dense vector cannot be empty")
		}
		values := append([]float32(nil), data.Dense.Values...)
		if err := validateFiniteFloat32(values); err != nil {
			return nil, fmt.Errorf("dense vector: %w", err)
		}
		return values, nil
	case *deepdatav3.VectorData_Sparse:
		if data.Sparse == nil || data.Sparse.Dim <= 0 {
			return nil, errors.New("sparse vector dimension must be positive")
		}
		if err := validateFiniteFloat32(data.Sparse.Values); err != nil {
			return nil, fmt.Errorf("sparse vector: %w", err)
		}
		return sparse.NewSparseVector(data.Sparse.Indices, data.Sparse.Values, int(data.Sparse.Dim))
	default:
		return nil, errors.New("vector data must contain dense or sparse values")
	}
}

func interfaceVectorsToProto(vectors map[string]interface{}) (map[string]*deepdatav3.VectorData, error) {
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

func vectorInterfaceToProto(value interface{}) (*deepdatav3.VectorData, error) {
	switch typed := value.(type) {
	case []float32:
		values := append([]float32(nil), typed...)
		if err := validateFiniteFloat32(values); err != nil {
			return nil, err
		}
		return denseVectorData(values), nil
	case []float64:
		values, err := float32Values(typed)
		if err != nil {
			return nil, err
		}
		return denseVectorData(values), nil
	case []interface{}:
		values, err := float32Values(typed)
		if err != nil {
			return nil, err
		}
		return denseVectorData(values), nil
	case *sparse.SparseVector:
		if typed == nil {
			return nil, errors.New("nil sparse vector")
		}
		return sparseVectorData(typed)
	case sparse.SparseVector:
		return sparseVectorData(&typed)
	case map[string]interface{}:
		indices, err := uint32Values(typed["indices"])
		if err != nil {
			return nil, fmt.Errorf("sparse indices: %w", err)
		}
		values, err := float32Values(typed["values"])
		if err != nil {
			return nil, fmt.Errorf("sparse values: %w", err)
		}
		dim, err := positiveInt32(typed["dim"])
		if err != nil {
			return nil, fmt.Errorf("sparse dimension: %w", err)
		}
		validated, err := sparse.NewSparseVector(indices, values, int(dim))
		if err != nil {
			return nil, err
		}
		return sparseVectorData(validated)
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

func float32Values(value interface{}) ([]float32, error) {
	switch typed := value.(type) {
	case []float32:
		values := append([]float32(nil), typed...)
		if err := validateFiniteFloat32(values); err != nil {
			return nil, err
		}
		return values, nil
	case []float64:
		values := make([]float32, len(typed))
		for i, number := range typed {
			if math.IsNaN(number) || math.IsInf(number, 0) || number > math.MaxFloat32 || number < -math.MaxFloat32 {
				return nil, fmt.Errorf("element %d is outside finite float32 range", i)
			}
			values[i] = float32(number)
		}
		return values, nil
	case []interface{}:
		values := make([]float32, len(typed))
		for i, item := range typed {
			number, ok := finiteFloat64(item)
			if !ok || number > math.MaxFloat32 || number < -math.MaxFloat32 {
				return nil, fmt.Errorf("element %d is not a finite float32 number", i)
			}
			values[i] = float32(number)
		}
		return values, nil
	default:
		return nil, fmt.Errorf("expected numeric array, got %T", value)
	}
}

func uint32Values(value interface{}) ([]uint32, error) {
	switch typed := value.(type) {
	case []uint32:
		return append([]uint32(nil), typed...), nil
	case []interface{}:
		values := make([]uint32, len(typed))
		for i, item := range typed {
			number, ok := finiteFloat64(item)
			if !ok || number < 0 || number > math.MaxUint32 || number != math.Trunc(number) {
				return nil, fmt.Errorf("element %d is not a uint32", i)
			}
			values[i] = uint32(number)
		}
		return values, nil
	default:
		return nil, fmt.Errorf("expected uint32 array, got %T", value)
	}
}

func positiveInt32(value interface{}) (int32, error) {
	number, ok := finiteFloat64(value)
	if !ok || number < 1 || number > math.MaxInt32 || number != math.Trunc(number) {
		return 0, fmt.Errorf("expected positive int32, got %v", value)
	}
	return int32(number), nil
}

func finiteFloat64(value interface{}) (float64, bool) {
	var number float64
	switch typed := value.(type) {
	case float64:
		number = typed
	case float32:
		number = float64(typed)
	case int:
		number = float64(typed)
	case int8:
		number = float64(typed)
	case int16:
		number = float64(typed)
	case int32:
		number = float64(typed)
	case int64:
		number = float64(typed)
	case uint:
		number = float64(typed)
	case uint8:
		number = float64(typed)
	case uint16:
		number = float64(typed)
	case uint32:
		number = float64(typed)
	case uint64:
		number = float64(typed)
	default:
		return 0, false
	}
	return number, !math.IsNaN(number) && !math.IsInf(number, 0)
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
		return status.Error(codes.InvalidArgument, "valid tenant_id required")
	}
	tenantCtx, _ := security.GetTenantContextFromContext(ctx)
	err := security.AuthorizeTenantAccess(tenantCtx, tenantID, collection, permission)
	if err == nil {
		return nil
	}
	if security.IsAuthorizationFailure(err, security.AuthorizationUnauthenticated) {
		return status.Error(codes.Unauthenticated, err.Error())
	}
	return status.Error(codes.PermissionDenied, err.Error())
}
