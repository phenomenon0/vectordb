package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Collection API Handlers

// handleCollectionCreate handles POST /admin/collection/create
func handleCollectionCreate(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse request body
		var config CollectionConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		// Validate tenant access (if multi-tenancy enabled)
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Create collection
		if err := store.CreateCollection(config); err != nil {
			http.Error(w, fmt.Sprintf("failed to create collection: %v", err), http.StatusBadRequest)
			return
		}

		// Return success response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "success",
			"message": fmt.Sprintf("collection %q created", config.Name),
		})
	}
}

// handleCollectionList handles GET /admin/collection/list
func handleCollectionList(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Get all collections
		collections := store.ListCollections()

		// Return collections
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":      "success",
			"count":       len(collections),
			"collections": collections,
		})
	}
}

// handleCollectionGet handles GET /admin/collection/{name}
func handleCollectionGet(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract collection name from URL
		path := strings.TrimPrefix(r.URL.Path, "/admin/collection/")
		collectionName := strings.Split(path, "/")[0]

		if collectionName == "" {
			http.Error(w, "collection name required", http.StatusBadRequest)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Get collection info
		info, err := store.GetCollectionInfo(collectionName)
		if err != nil {
			http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
			return
		}

		// Return collection info
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     "success",
			"collection": info,
		})
	}
}

// handleCollectionStats handles GET /admin/collection/{name}/stats
func handleCollectionStats(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract collection name from URL
		path := strings.TrimPrefix(r.URL.Path, "/admin/collection/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 {
			http.Error(w, "invalid URL format", http.StatusBadRequest)
			return
		}
		collectionName := parts[0]

		if collectionName == "" {
			http.Error(w, "collection name required", http.StatusBadRequest)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Get collection stats
		info, err := store.GetCollectionInfo(collectionName)
		if err != nil {
			http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
			return
		}

		// Return stats
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "success",
			"name":   collectionName,
			"stats":  info.IndexStats,
		})
	}
}

// handleCollectionUpdate handles PUT /admin/collection/{name}/config
func handleCollectionUpdate(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract collection name from URL
		path := strings.TrimPrefix(r.URL.Path, "/admin/collection/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 {
			http.Error(w, "invalid URL format", http.StatusBadRequest)
			return
		}
		collectionName := parts[0]

		if collectionName == "" {
			http.Error(w, "collection name required", http.StatusBadRequest)
			return
		}

		// Parse new config
		var config map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Update collection config
		if err := store.UpdateCollectionConfig(collectionName, config); err != nil {
			http.Error(w, fmt.Sprintf("failed to update collection: %v", err), http.StatusBadRequest)
			return
		}

		// Return success
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "success",
			"message": fmt.Sprintf("collection %q updated", collectionName),
		})
	}
}

// handleCollectionDelete handles DELETE /admin/collection/{name}
func handleCollectionDelete(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract collection name from URL
		path := strings.TrimPrefix(r.URL.Path, "/admin/collection/")
		collectionName := strings.Split(path, "/")[0]

		if collectionName == "" {
			http.Error(w, "collection name required", http.StatusBadRequest)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Delete collection
		if err := store.DeleteCollection(collectionName); err != nil {
			http.Error(w, fmt.Sprintf("failed to delete collection: %v", err), http.StatusBadRequest)
			return
		}

		// Return success
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "success",
			"message": fmt.Sprintf("collection %q deleted", collectionName),
		})
	}
}

// handleAllCollectionStats handles GET /admin/collection/stats (all collections)
func handleAllCollectionStats(store *VectorStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Validate tenant access
		tenantCtx, ok := GetTenantContextFromContext(r.Context())
		if !ok && store.requireAuth {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check admin permissions
		if tenantCtx != nil && !tenantCtx.IsAdmin {
			http.Error(w, "admin permission required", http.StatusForbidden)
			return
		}

		// Get all collection stats
		stats := store.GetAllCollectionStats()

		// Return stats
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "success",
			"stats":  stats,
		})
	}
}
