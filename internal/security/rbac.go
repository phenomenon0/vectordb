package security

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

// ===========================================================================================
// ROLE-BASED ACCESS CONTROL (RBAC)
// Fine-grained permission system with predefined roles and custom role support
// ===========================================================================================

// Permission represents a specific action that can be performed
type Permission string

const (
	// Vector operations
	PermVectorInsert Permission = "vector:insert"
	PermVectorQuery  Permission = "vector:query"
	PermVectorDelete Permission = "vector:delete"
	PermVectorUpdate Permission = "vector:update"

	// Collection operations
	PermCollectionCreate Permission = "collection:create"
	PermCollectionDelete Permission = "collection:delete"
	PermCollectionList   Permission = "collection:list"
	PermCollectionInfo   Permission = "collection:info"

	// Admin operations
	PermAdminUsers     Permission = "admin:users"
	PermAdminRoles     Permission = "admin:roles"
	PermAdminCluster   Permission = "admin:cluster"
	PermAdminBackup    Permission = "admin:backup"
	PermAdminRestore   Permission = "admin:restore"
	PermAdminMetrics   Permission = "admin:metrics"
	PermAdminAuditLogs Permission = "admin:audit_logs"

	// System operations
	PermSystemShutdown Permission = "system:shutdown"
	PermSystemConfig   Permission = "system:config"

	// Wildcard permissions
	PermVectorAll     Permission = "vector:*"
	PermCollectionAll Permission = "collection:*"
	PermAdminAll      Permission = "admin:*"
	PermAll           Permission = "*"
)

// Role represents a named set of permissions
type Role struct {
	Name        string       `json:"name"`
	Description string       `json:"description"`
	Permissions []Permission `json:"permissions"`
	CreatedAt   time.Time    `json:"created_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
	IsBuiltIn   bool         `json:"is_built_in"` // Built-in roles can't be deleted
}

// PredefinedRoles returns the built-in roles
func PredefinedRoles() map[string]*Role {
	return map[string]*Role{
		"viewer": {
			Name:        "viewer",
			Description: "Read-only access to vectors and collections",
			Permissions: []Permission{
				PermVectorQuery,
				PermCollectionList,
				PermCollectionInfo,
			},
			IsBuiltIn: true,
		},
		"editor": {
			Name:        "editor",
			Description: "Can read and write vectors, manage collections",
			Permissions: []Permission{
				PermVectorInsert,
				PermVectorQuery,
				PermVectorDelete,
				PermVectorUpdate,
				PermCollectionCreate,
				PermCollectionDelete,
				PermCollectionList,
				PermCollectionInfo,
			},
			IsBuiltIn: true,
		},
		"operator": {
			Name:        "operator",
			Description: "Editor plus backup/restore and metrics access",
			Permissions: []Permission{
				PermVectorAll,
				PermCollectionAll,
				PermAdminBackup,
				PermAdminRestore,
				PermAdminMetrics,
			},
			IsBuiltIn: true,
		},
		"admin": {
			Name:        "admin",
			Description: "Full administrative access",
			Permissions: []Permission{
				PermAll,
			},
			IsBuiltIn: true,
		},
	}
}

// RoleAssignment assigns a role to a user/tenant with optional scope
type RoleAssignment struct {
	ID          string     `json:"id"`
	SubjectType string     `json:"subject_type"` // "user" or "tenant"
	SubjectID   string     `json:"subject_id"`   // User ID or Tenant ID
	RoleName    string     `json:"role_name"`
	Scope       *Scope     `json:"scope,omitempty"` // Optional: limit to specific resources
	GrantedBy   string     `json:"granted_by"`
	GrantedAt   time.Time  `json:"granted_at"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
}

// Scope limits a role assignment to specific resources
type Scope struct {
	Collections []string `json:"collections,omitempty"` // Specific collections (empty = all)
	Tenants     []string `json:"tenants,omitempty"`     // For admin roles managing specific tenants
}

// RBACManager manages roles and role assignments
type RBACManager struct {
	mu          sync.RWMutex
	roles       map[string]*Role             // role name -> Role
	assignments map[string][]*RoleAssignment // subject ID -> []RoleAssignment
	permCache   map[string]*permissionSet    // subject ID -> cached permissions
	persistPath string
}

// permissionSet is a cached set of permissions for fast lookup
type permissionSet struct {
	permissions map[Permission]bool
	collections map[string]bool // allowed collections (empty = all)
	expiry      time.Time
}

// NewRBACManager creates a new RBAC manager
func NewRBACManager(persistPath string) *RBACManager {
	rm := &RBACManager{
		roles:       make(map[string]*Role),
		assignments: make(map[string][]*RoleAssignment),
		permCache:   make(map[string]*permissionSet),
		persistPath: persistPath,
	}

	// Load predefined roles
	for name, role := range PredefinedRoles() {
		role.CreatedAt = time.Now()
		role.UpdatedAt = time.Now()
		rm.roles[name] = role
	}

	// Try to load persisted data
	if persistPath != "" {
		rm.loadFromFile()
	}

	return rm
}

// ===========================================================================================
// ROLE MANAGEMENT
// ===========================================================================================

// CreateRole creates a new custom role
func (rm *RBACManager) CreateRole(name, description string, permissions []Permission) (*Role, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if _, exists := rm.roles[name]; exists {
		return nil, fmt.Errorf("role %s already exists", name)
	}

	role := &Role{
		Name:        name,
		Description: description,
		Permissions: permissions,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		IsBuiltIn:   false,
	}

	rm.roles[name] = role
	rm.invalidateCache()
	rm.persist()

	return role, nil
}

// UpdateRole updates an existing role's permissions
func (rm *RBACManager) UpdateRole(name string, permissions []Permission) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	role, exists := rm.roles[name]
	if !exists {
		return fmt.Errorf("role %s not found", name)
	}

	if role.IsBuiltIn {
		return fmt.Errorf("cannot modify built-in role %s", name)
	}

	role.Permissions = permissions
	role.UpdatedAt = time.Now()
	rm.invalidateCache()
	rm.persist()

	return nil
}

// DeleteRole deletes a custom role
func (rm *RBACManager) DeleteRole(name string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	role, exists := rm.roles[name]
	if !exists {
		return fmt.Errorf("role %s not found", name)
	}

	if role.IsBuiltIn {
		return fmt.Errorf("cannot delete built-in role %s", name)
	}

	delete(rm.roles, name)
	rm.invalidateCache()
	rm.persist()

	return nil
}

// GetRole returns a role by name
func (rm *RBACManager) GetRole(name string) (*Role, bool) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	role, exists := rm.roles[name]
	return role, exists
}

// ListRoles returns all roles
func (rm *RBACManager) ListRoles() []*Role {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	roles := make([]*Role, 0, len(rm.roles))
	for _, role := range rm.roles {
		roles = append(roles, role)
	}
	return roles
}

// ===========================================================================================
// ROLE ASSIGNMENT
// ===========================================================================================

// AssignRole assigns a role to a subject (user or tenant)
func (rm *RBACManager) AssignRole(subjectType, subjectID, roleName, grantedBy string, scope *Scope, expiresIn *time.Duration) (*RoleAssignment, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	// Verify role exists
	if _, exists := rm.roles[roleName]; !exists {
		return nil, fmt.Errorf("role %s not found", roleName)
	}

	// Check for duplicate assignment
	for _, existing := range rm.assignments[subjectID] {
		if existing.RoleName == roleName && scopesEqual(existing.Scope, scope) {
			return nil, fmt.Errorf("role %s already assigned to %s", roleName, subjectID)
		}
	}

	var expiresAt *time.Time
	if expiresIn != nil {
		t := time.Now().Add(*expiresIn)
		expiresAt = &t
	}

	assignment := &RoleAssignment{
		ID:          fmt.Sprintf("%s_%s_%d", subjectID, roleName, time.Now().UnixNano()),
		SubjectType: subjectType,
		SubjectID:   subjectID,
		RoleName:    roleName,
		Scope:       scope,
		GrantedBy:   grantedBy,
		GrantedAt:   time.Now(),
		ExpiresAt:   expiresAt,
	}

	rm.assignments[subjectID] = append(rm.assignments[subjectID], assignment)
	rm.invalidateCacheFor(subjectID)
	rm.persist()

	return assignment, nil
}

// RevokeRole removes a role assignment
func (rm *RBACManager) RevokeRole(subjectID, roleName string, scope *Scope) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	assignments := rm.assignments[subjectID]
	for i, a := range assignments {
		if a.RoleName == roleName && scopesEqual(a.Scope, scope) {
			rm.assignments[subjectID] = append(assignments[:i], assignments[i+1:]...)
			rm.invalidateCacheFor(subjectID)
			rm.persist()
			return nil
		}
	}

	return fmt.Errorf("role assignment not found")
}

// GetAssignments returns all role assignments for a subject
func (rm *RBACManager) GetAssignments(subjectID string) []*RoleAssignment {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	// Filter out expired assignments
	var valid []*RoleAssignment
	for _, a := range rm.assignments[subjectID] {
		if a.ExpiresAt == nil || time.Now().Before(*a.ExpiresAt) {
			valid = append(valid, a)
		}
	}
	return valid
}

// ===========================================================================================
// PERMISSION CHECKING
// ===========================================================================================

// HasPermission checks if a subject has a specific permission
func (rm *RBACManager) HasPermission(subjectID string, permission Permission, collection string) bool {
	rm.mu.RLock()

	// Check cache first
	cached, ok := rm.permCache[subjectID]
	if ok && time.Now().Before(cached.expiry) {
		result := rm.checkCachedPermission(cached, permission, collection)
		rm.mu.RUnlock()
		return result
	}
	rm.mu.RUnlock()

	// Build permission set
	rm.mu.Lock()
	permSet := rm.buildPermissionSet(subjectID)
	rm.permCache[subjectID] = permSet
	rm.mu.Unlock()

	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.checkCachedPermission(permSet, permission, collection)
}

// HasAnyPermission checks if a subject has any of the given permissions
func (rm *RBACManager) HasAnyPermission(subjectID string, permissions []Permission, collection string) bool {
	for _, perm := range permissions {
		if rm.HasPermission(subjectID, perm, collection) {
			return true
		}
	}
	return false
}

// HasAllPermissions checks if a subject has all of the given permissions
func (rm *RBACManager) HasAllPermissions(subjectID string, permissions []Permission, collection string) bool {
	for _, perm := range permissions {
		if !rm.HasPermission(subjectID, perm, collection) {
			return false
		}
	}
	return true
}

// buildPermissionSet builds a complete permission set for a subject
func (rm *RBACManager) buildPermissionSet(subjectID string) *permissionSet {
	permSet := &permissionSet{
		permissions: make(map[Permission]bool),
		collections: make(map[string]bool),
		expiry:      time.Now().Add(5 * time.Minute), // Cache for 5 minutes
	}

	for _, assignment := range rm.assignments[subjectID] {
		// Skip expired assignments
		if assignment.ExpiresAt != nil && time.Now().After(*assignment.ExpiresAt) {
			continue
		}

		role := rm.roles[assignment.RoleName]
		if role == nil {
			continue
		}

		// Add permissions from role
		for _, perm := range role.Permissions {
			permSet.permissions[perm] = true
		}

		// Add scoped collections
		if assignment.Scope != nil {
			for _, coll := range assignment.Scope.Collections {
				permSet.collections[coll] = true
			}
		}
	}

	return permSet
}

// checkCachedPermission checks a permission against the cached set
func (rm *RBACManager) checkCachedPermission(cached *permissionSet, permission Permission, collection string) bool {
	// Check wildcard permissions first
	if cached.permissions[PermAll] {
		return true
	}

	// Check category wildcard (e.g., vector:* for vector:insert)
	category := getPermissionCategory(permission)
	if cached.permissions[Permission(category+":*")] {
		return true
	}

	// Check specific permission
	if !cached.permissions[permission] {
		return false
	}

	// Check collection scope (if any collections are specified, user must have access to this one)
	if collection != "" && len(cached.collections) > 0 {
		return cached.collections[collection]
	}

	return true
}

// getPermissionCategory returns the category part of a permission (e.g., "vector" from "vector:insert")
func getPermissionCategory(perm Permission) string {
	s := string(perm)
	for i, c := range s {
		if c == ':' {
			return s[:i]
		}
	}
	return s
}

// ===========================================================================================
// HELPER FUNCTIONS
// ===========================================================================================

func scopesEqual(a, b *Scope) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	if len(a.Collections) != len(b.Collections) {
		return false
	}
	for i, c := range a.Collections {
		if b.Collections[i] != c {
			return false
		}
	}

	if len(a.Tenants) != len(b.Tenants) {
		return false
	}
	for i, t := range a.Tenants {
		if b.Tenants[i] != t {
			return false
		}
	}

	return true
}

func (rm *RBACManager) invalidateCache() {
	rm.permCache = make(map[string]*permissionSet)
}

func (rm *RBACManager) invalidateCacheFor(subjectID string) {
	delete(rm.permCache, subjectID)
}

// ===========================================================================================
// PERSISTENCE
// ===========================================================================================

type rbacPersistData struct {
	CustomRoles map[string]*Role             `json:"custom_roles"`
	Assignments map[string][]*RoleAssignment `json:"assignments"`
}

func (rm *RBACManager) persist() {
	if rm.persistPath == "" {
		return
	}

	// Collect custom roles only
	customRoles := make(map[string]*Role)
	for name, role := range rm.roles {
		if !role.IsBuiltIn {
			customRoles[name] = role
		}
	}

	data := rbacPersistData{
		CustomRoles: customRoles,
		Assignments: rm.assignments,
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		fmt.Printf("[RBAC] Failed to marshal data: %v\n", err)
		return
	}

	if err := os.WriteFile(rm.persistPath, jsonData, 0600); err != nil {
		fmt.Printf("[RBAC] Failed to save data: %v\n", err)
	}
}

func (rm *RBACManager) loadFromFile() {
	data, err := os.ReadFile(rm.persistPath)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Printf("[RBAC] Failed to load data: %v\n", err)
		}
		return
	}

	var persistData rbacPersistData
	if err := json.Unmarshal(data, &persistData); err != nil {
		fmt.Printf("[RBAC] Failed to parse data: %v\n", err)
		return
	}

	// Load custom roles
	for name, role := range persistData.CustomRoles {
		rm.roles[name] = role
	}

	// Load assignments
	rm.assignments = persistData.Assignments
	if rm.assignments == nil {
		rm.assignments = make(map[string][]*RoleAssignment)
	}

	fmt.Printf("[RBAC] Loaded %d custom roles, %d subjects with assignments\n",
		len(persistData.CustomRoles), len(rm.assignments))
}

// ===========================================================================================
// HTTP HANDLERS FOR RBAC MANAGEMENT
// ===========================================================================================

// RBACHandlers provides HTTP handlers for RBAC management
type RBACHandlers struct {
	rbac *RBACManager
}

// NewRBACHandlers creates RBAC HTTP handlers
func NewRBACHandlers(rbac *RBACManager) *RBACHandlers {
	return &RBACHandlers{rbac: rbac}
}

// HandleListRoles handles GET /admin/rbac/roles
func (h *RBACHandlers) HandleListRoles(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	roles := h.rbac.ListRoles()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"roles": roles,
		"count": len(roles),
	})
}

// HandleCreateRole handles POST /admin/rbac/roles
func (h *RBACHandlers) HandleCreateRole(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Name        string   `json:"name"`
		Description string   `json:"description"`
		Permissions []string `json:"permissions"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	perms := make([]Permission, len(req.Permissions))
	for i, p := range req.Permissions {
		perms[i] = Permission(p)
	}

	role, err := h.rbac.CreateRole(req.Name, req.Description, perms)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create role: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(role)
}

// HandleAssignRole handles POST /admin/rbac/assign
func (h *RBACHandlers) HandleAssignRole(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SubjectType string   `json:"subject_type"` // "user" or "tenant"
		SubjectID   string   `json:"subject_id"`
		RoleName    string   `json:"role_name"`
		Collections []string `json:"collections,omitempty"`
		ExpiresIn   int      `json:"expires_in_hours,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	var scope *Scope
	if len(req.Collections) > 0 {
		scope = &Scope{Collections: req.Collections}
	}

	var expiresIn *time.Duration
	if req.ExpiresIn > 0 {
		d := time.Duration(req.ExpiresIn) * time.Hour
		expiresIn = &d
	}

	// Get granter from request header or default to system
	grantedBy := r.Header.Get("X-Granted-By")
	if grantedBy == "" {
		grantedBy = "system"
	}

	assignment, err := h.rbac.AssignRole(req.SubjectType, req.SubjectID, req.RoleName, grantedBy, scope, expiresIn)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to assign role: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(assignment)
}

// HandleRevokeRole handles POST /admin/rbac/revoke
func (h *RBACHandlers) HandleRevokeRole(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SubjectID   string   `json:"subject_id"`
		RoleName    string   `json:"role_name"`
		Collections []string `json:"collections,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	var scope *Scope
	if len(req.Collections) > 0 {
		scope = &Scope{Collections: req.Collections}
	}

	if err := h.rbac.RevokeRole(req.SubjectID, req.RoleName, scope); err != nil {
		http.Error(w, fmt.Sprintf("failed to revoke role: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "revoked"})
}

// HandleGetAssignments handles GET /admin/rbac/assignments?subject_id=xxx
func (h *RBACHandlers) HandleGetAssignments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	subjectID := r.URL.Query().Get("subject_id")
	if subjectID == "" {
		http.Error(w, "subject_id required", http.StatusBadRequest)
		return
	}

	assignments := h.rbac.GetAssignments(subjectID)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"subject_id":  subjectID,
		"assignments": assignments,
		"count":       len(assignments),
	})
}

// HandleCheckPermission handles POST /admin/rbac/check
func (h *RBACHandlers) HandleCheckPermission(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SubjectID  string `json:"subject_id"`
		Permission string `json:"permission"`
		Collection string `json:"collection,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	allowed := h.rbac.HasPermission(req.SubjectID, Permission(req.Permission), req.Collection)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"subject_id": req.SubjectID,
		"permission": req.Permission,
		"collection": req.Collection,
		"allowed":    allowed,
	})
}
