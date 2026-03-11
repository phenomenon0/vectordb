package main

import (
	"context"
	"fmt"

	deepdatav1 "github.com/phenomenon0/vectordb/api/gen/deepdata/v1"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// CollectionGRPCServer implements the DeepData gRPC service.
// It delegates to the same CollectionManager used by the HTTP server.
type CollectionGRPCServer struct {
	deepdatav1.UnimplementedDeepDataServer
	manager *vcollection.CollectionManager
}

func (s *CollectionGRPCServer) CreateCollection(ctx context.Context, req *deepdatav1.CreateCollectionRequest) (*deepdatav1.CreateCollectionResponse, error) {
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "collection name required")
	}

	fields := make([]vcollection.VectorField, len(req.Fields))
	for i, f := range req.Fields {
		idxType, err := parseIndexType(f.IndexType)
		if err != nil {
			idxType = vcollection.IndexTypeHNSW // default
		}

		params := make(map[string]interface{}, len(f.IndexParams))
		for k, v := range f.IndexParams {
			params[k] = v
		}

		fields[i] = vcollection.VectorField{
			Name: f.Name,
			Type: vcollection.VectorType(f.Type),
			Dim:  int(f.Dim),
			Index: vcollection.IndexConfig{
				Type:   idxType,
				Params: params,
			},
		}
	}

	schema := vcollection.CollectionSchema{
		Name:   req.Name,
		Fields: fields,
	}

	if _, err := s.manager.CreateCollection(ctx, schema); err != nil {
		return nil, status.Errorf(codes.AlreadyExists, "%v", err)
	}

	return &deepdatav1.CreateCollectionResponse{Name: req.Name}, nil
}

func (s *CollectionGRPCServer) DeleteCollection(ctx context.Context, req *deepdatav1.DeleteCollectionRequest) (*deepdatav1.DeleteCollectionResponse, error) {
	if err := s.manager.DeleteCollection(ctx, req.Name); err != nil {
		return nil, status.Errorf(codes.NotFound, "%v", err)
	}
	return &deepdatav1.DeleteCollectionResponse{}, nil
}

func (s *CollectionGRPCServer) Insert(ctx context.Context, req *deepdatav1.InsertRequest) (*deepdatav1.InsertResponse, error) {
	if req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}

	vectors := make(map[string]interface{}, len(req.Vectors))
	for name, vd := range req.Vectors {
		v, err := vectorDataToInterface(vd)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "field %s: %v", name, err)
		}
		vectors[name] = v
	}

	meta := protoMetaToInterface(req.Metadata)

	doc := &vcollection.Document{
		Vectors:  vectors,
		Metadata: meta,
	}

	if err := s.manager.AddDocument(ctx, req.Collection, doc); err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}

	return &deepdatav1.InsertResponse{Id: doc.ID}, nil
}

func (s *CollectionGRPCServer) BatchInsert(ctx context.Context, req *deepdatav1.BatchInsertRequest) (*deepdatav1.BatchInsertResponse, error) {
	if req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}

	docs := make([]vcollection.Document, len(req.Docs))
	for i, bd := range req.Docs {
		vectors := make(map[string]interface{}, len(bd.Vectors))
		for name, vd := range bd.Vectors {
			v, err := vectorDataToInterface(vd)
			if err != nil {
				return nil, status.Errorf(codes.InvalidArgument, "doc %d field %s: %v", i, name, err)
			}
			vectors[name] = v
		}
		docs[i] = vcollection.Document{
			Vectors:  vectors,
			Metadata: protoMetaToInterface(bd.Metadata),
		}
	}

	if err := s.manager.BatchAddDocuments(ctx, req.Collection, docs); err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}

	ids := make([]uint64, len(docs))
	for i := range docs {
		ids[i] = docs[i].ID
	}

	return &deepdatav1.BatchInsertResponse{
		Ids:      ids,
		Inserted: int32(len(docs)),
	}, nil
}

func (s *CollectionGRPCServer) Search(ctx context.Context, req *deepdatav1.SearchRequest) (*deepdatav1.SearchResponse, error) {
	if req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if req.TopK <= 0 {
		return nil, status.Error(codes.InvalidArgument, "top_k must be positive")
	}

	queries := make(map[string]interface{}, len(req.Queries))
	for name, vd := range req.Queries {
		v, err := vectorDataToInterface(vd)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "query %s: %v", name, err)
		}
		queries[name] = v
	}

	// Convert simplified eq-filter to the format filter.FromMap expects
	var filters map[string]interface{}
	if len(req.MetadataFilter) > 0 {
		filters = make(map[string]interface{}, len(req.MetadataFilter))
		for k, v := range req.MetadataFilter {
			filters[k] = map[string]interface{}{"$eq": v}
		}
	}

	searchReq := vcollection.SearchRequest{
		CollectionName: req.Collection,
		Queries:        queries,
		TopK:           int(req.TopK),
		EfSearch:       int(req.EfSearch),
		Filters:        filters,
	}
	if req.IncludeVectors {
		t := true
		searchReq.IncludeVectors = &t
	}

	resp, err := s.manager.SearchCollection(ctx, searchReq)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}

	hits := make([]*deepdatav1.SearchHit, len(resp.Documents))
	for i, doc := range resp.Documents {
		hit := &deepdatav1.SearchHit{
			Id: doc.ID,
		}
		if i < len(resp.Scores) {
			hit.Score = resp.Scores[i]
		}
		if doc.Metadata != nil {
			hit.Metadata = make(map[string]string, len(doc.Metadata))
			for k, v := range doc.Metadata {
				hit.Metadata[k] = fmt.Sprintf("%v", v)
			}
		}
		hits[i] = hit
	}

	return &deepdatav1.SearchResponse{
		Results:            hits,
		CandidatesExamined: int32(resp.CandidatesExamined),
	}, nil
}

func (s *CollectionGRPCServer) DeleteDoc(ctx context.Context, req *deepdatav1.DeleteDocRequest) (*deepdatav1.DeleteDocResponse, error) {
	if err := s.manager.DeleteDocument(ctx, req.Collection, req.DocId); err != nil {
		return nil, status.Errorf(codes.NotFound, "%v", err)
	}
	return &deepdatav1.DeleteDocResponse{}, nil
}

func (s *CollectionGRPCServer) Recommend(ctx context.Context, req *deepdatav1.RecommendRequest) (*deepdatav1.SearchResponse, error) {
	if req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if len(req.PositiveIds) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one positive_id required")
	}

	topK := int(req.TopK)
	if topK <= 0 {
		topK = 10
	}

	var filters map[string]interface{}
	if len(req.MetadataFilter) > 0 {
		filters = make(map[string]interface{}, len(req.MetadataFilter))
		for k, v := range req.MetadataFilter {
			filters[k] = map[string]interface{}{"$eq": v}
		}
	}

	recReq := vcollection.RecommendRequest{
		CollectionName: req.Collection,
		FieldName:      req.Field,
		PositiveIDs:    req.PositiveIds,
		NegativeIDs:    req.NegativeIds,
		NegativeWeight: req.NegativeWeight,
		TopK:           topK,
		EfSearch:       int(req.EfSearch),
		Filters:        filters,
	}

	resp, err := s.manager.Recommend(ctx, recReq)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}

	return searchResponseToProto(resp), nil
}

func (s *CollectionGRPCServer) Discover(ctx context.Context, req *deepdatav1.DiscoverRequest) (*deepdatav1.SearchResponse, error) {
	if req.Collection == "" {
		return nil, status.Error(codes.InvalidArgument, "collection required")
	}
	if len(req.Context) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one context pair required")
	}

	topK := int(req.TopK)
	if topK <= 0 {
		topK = 10
	}

	var filters map[string]interface{}
	if len(req.MetadataFilter) > 0 {
		filters = make(map[string]interface{}, len(req.MetadataFilter))
		for k, v := range req.MetadataFilter {
			filters[k] = map[string]interface{}{"$eq": v}
		}
	}

	context := make([]vcollection.ContextPair, len(req.Context))
	for i, c := range req.Context {
		context[i] = vcollection.ContextPair{
			PositiveID: c.PositiveId,
			NegativeID: c.NegativeId,
		}
	}

	var targetVec []float32
	if req.TargetVector != nil {
		targetVec = req.TargetVector.Values
	}

	discReq := vcollection.DiscoverRequest{
		CollectionName: req.Collection,
		FieldName:      req.Field,
		TargetID:       req.TargetId,
		TargetVector:   targetVec,
		Context:        context,
		TopK:           topK,
		EfSearch:       int(req.EfSearch),
		Filters:        filters,
	}

	resp, err := s.manager.Discover(ctx, discReq)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "%v", err)
	}

	return searchResponseToProto(resp), nil
}

func searchResponseToProto(resp *vcollection.SearchResponse) *deepdatav1.SearchResponse {
	hits := make([]*deepdatav1.SearchHit, len(resp.Documents))
	for i, doc := range resp.Documents {
		hit := &deepdatav1.SearchHit{
			Id: doc.ID,
		}
		if i < len(resp.Scores) {
			hit.Score = resp.Scores[i]
		}
		if doc.Metadata != nil {
			hit.Metadata = make(map[string]string, len(doc.Metadata))
			for k, v := range doc.Metadata {
				hit.Metadata[k] = fmt.Sprintf("%v", v)
			}
		}
		hits[i] = hit
	}
	return &deepdatav1.SearchResponse{
		Results:            hits,
		CandidatesExamined: int32(resp.CandidatesExamined),
	}
}

// ── Helpers ──────────────────────────────────────────────────────────

func vectorDataToInterface(vd *deepdatav1.VectorData) (interface{}, error) {
	if vd == nil {
		return nil, fmt.Errorf("nil vector data")
	}
	switch d := vd.Data.(type) {
	case *deepdatav1.VectorData_Dense:
		// Return the float slice directly — zero-copy from protobuf decode.
		// This is the key latency win over JSON.
		return d.Dense.Values, nil
	case *deepdatav1.VectorData_Sparse:
		return map[string]interface{}{
			"indices": d.Sparse.Indices,
			"values":  d.Sparse.Values,
			"dim":     int(d.Sparse.Dim),
		}, nil
	default:
		return nil, fmt.Errorf("unknown vector data type")
	}
}

func protoMetaToInterface(meta map[string]string) map[string]interface{} {
	if len(meta) == 0 {
		return nil
	}
	out := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		out[k] = v
	}
	return out
}

func parseIndexType(s string) (vcollection.IndexType, error) {
	if s == "" {
		return vcollection.IndexTypeHNSW, nil
	}
	return vcollection.ParseIndexType(s)
}
