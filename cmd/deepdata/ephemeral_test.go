package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// An agent picks the durability class at create time, so the class it asked for
// has to be readable back from the collection it created — otherwise nothing
// tells it whether its documents will survive a restart.
func TestEphemeralDurabilityRoundTripsOverHTTP(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	body := `{"name":"frames","durability":"ephemeral","fields":[{"name":"embedding","type":"dense","dim":4,"index":{"type":"flat"}}]}`
	req := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", resp.Code, resp.Body.String())
	}

	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/frames", nil))
	if get.Code != http.StatusOK {
		t.Fatalf("get returned %d: %s", get.Code, get.Body.String())
	}
	var decoded struct {
		Collection struct {
			Durability string `json:"durability"`
		} `json:"collection"`
	}
	if err := json.Unmarshal(get.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Collection.Durability != vcollection.DurabilityEphemeral {
		t.Fatalf("durability = %q, want %q", decoded.Collection.Durability, vcollection.DurabilityEphemeral)
	}
}

// A collection created without the field is durable, and the read says so
// rather than leaving the caller to guess what an absent value means.
func TestEphemeralDefaultDurabilityIsReportedOverHTTP(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	body := `{"name":"kept","fields":[{"name":"embedding","type":"dense","dim":4,"index":{"type":"flat"}}]}`
	req := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", resp.Code, resp.Body.String())
	}

	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/kept", nil))
	var decoded struct {
		Collection struct {
			Durability string `json:"durability"`
		} `json:"collection"`
	}
	if err := json.Unmarshal(get.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Collection.Durability != vcollection.DurabilityDurable {
		t.Fatalf("default durability = %q, want %q", decoded.Collection.Durability, vcollection.DurabilityDurable)
	}
}

// Capabilities are how a client discovers what this build accepts before it
// sends a create it cannot know will be rejected.
func TestEphemeralDurabilityClassesAreAdvertisedByStatus(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/v3/status", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("status returned %d: %s", resp.Code, resp.Body.String())
	}
	var decoded struct {
		Capabilities struct {
			DurabilityClasses []string `json:"durability_classes"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	want := []string{vcollection.DurabilityDurable, vcollection.DurabilityEphemeral}
	if len(decoded.Capabilities.DurabilityClasses) != len(want) {
		t.Fatalf("durability_classes = %v, want %v", decoded.Capabilities.DurabilityClasses, want)
	}
	for i, class := range want {
		if decoded.Capabilities.DurabilityClasses[i] != class {
			t.Fatalf("durability_classes = %v, want %v", decoded.Capabilities.DurabilityClasses, want)
		}
	}
}

// The four transports must agree on the class, so gRPC carries it on the
// request and back on CollectionInfo exactly as HTTP does.
func TestEphemeralDurabilityRoundTripsOverGRPC(t *testing.T) {
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

	for name, durability := range map[string]string{"frames": vcollection.DurabilityEphemeral, "kept": ""} {
		if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
			TenantId:   "acme",
			Name:       name,
			Durability: durability,
			Fields: []*deepdatav3.VectorFieldConfig{
				{Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 4, IndexType: "flat"},
			},
		}); err != nil {
			t.Fatalf("create %s: %v", name, err)
		}
	}

	assertDurability := func(name, want string) {
		t.Helper()
		got, err := server.GetCollection(ctx, &deepdatav3.GetCollectionRequest{TenantId: "acme", Name: name})
		if err != nil {
			t.Fatalf("get %s: %v", name, err)
		}
		if got.Collection.GetDurability() != want {
			t.Fatalf("%s durability = %q, want %q", name, got.Collection.GetDurability(), want)
		}
	}
	assertDurability("frames", vcollection.DurabilityEphemeral)
	assertDurability("kept", vcollection.DurabilityDurable)

	if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId:   "acme",
		Name:       "bogus",
		Durability: "temporary",
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "embedding", Type: int32(vcollection.VectorTypeDense), Dim: 4, IndexType: "flat"},
		},
	}); err == nil {
		t.Fatal("gRPC accepted an unknown durability class")
	}
}
