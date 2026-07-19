package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

func failf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "container gRPC probe: "+format+"\n", args...)
	os.Exit(1)
}

func dense(values ...float32) *deepdatav3.VectorData {
	return &deepdatav3.VectorData{Data: &deepdatav3.VectorData_Dense{
		Dense: &deepdatav3.DenseVector{Values: values},
	}}
}

func containsID(results []*deepdatav3.SearchHit, id uint64) bool {
	for _, result := range results {
		if result.GetId() == id {
			return true
		}
	}
	return false
}

func main() {
	if len(os.Args) != 6 {
		failf("usage: probe address token tenant collection document-id")
	}

	address, token := os.Args[1], os.Args[2]
	tenant, collection := os.Args[3], os.Args[4]
	documentID, err := strconv.ParseUint(os.Args[5], 10, 64)
	if err != nil || documentID == 0 {
		failf("invalid document ID %q", os.Args[5])
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(
		ctx,
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		failf("connect to %s: %v", address, err)
	}
	defer conn.Close()

	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+token)
	client := deepdatav3.NewDeepDataClient(conn)

	info, err := client.GetTenantInfo(ctx, &deepdatav3.GetTenantInfoRequest{TenantId: tenant})
	if err != nil {
		failf("authenticated GetTenantInfo: %v", err)
	}
	if info.GetTenantId() != tenant || info.GetCollectionCount() != 1 || info.GetTotalDocuments() != 1 {
		failf("unexpected tenant state: %+v", info)
	}

	listed, err := client.ListCollections(ctx, &deepdatav3.ListCollectionsRequest{TenantId: tenant})
	if err != nil {
		failf("authenticated ListCollections: %v", err)
	}
	if len(listed.GetCollections()) != 1 || listed.GetCollections()[0].GetName() != collection {
		failf("unexpected collection list: %+v", listed.GetCollections())
	}

	got, err := client.GetCollection(ctx, &deepdatav3.GetCollectionRequest{
		TenantId: tenant,
		Name:     collection,
	})
	if err != nil {
		failf("authenticated GetCollection: %v", err)
	}
	if got.GetCollection().GetName() != collection || got.GetCollection().GetDocumentCount() != 1 {
		failf("unexpected collection state: %+v", got.GetCollection())
	}

	search, err := client.Search(ctx, &deepdatav3.SearchRequest{
		TenantId:   tenant,
		Collection: collection,
		TopK:       10,
		Queries: map[string]*deepdatav3.VectorData{
			"embedding": dense(1, 0, 0),
		},
	})
	if err != nil {
		failf("authenticated Search: %v", err)
	}
	if !containsID(search.GetResults(), documentID) {
		failf("search did not return document %d", documentID)
	}

	fmt.Printf("authenticated deepdata.v3 state verified for %s/%s document %d\n", tenant, collection, documentID)
}
