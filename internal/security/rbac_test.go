package security

import (
	"testing"
	"time"
)

// TestPredefinedRoles tests that predefined roles exist with correct permissions
func TestPredefinedRoles(t *testing.T) {
	roles := PredefinedRoles()

	testCases := []struct {
		name     string
		expected int // minimum permissions
	}{
		{"viewer", 3},
		{"editor", 8},
		{"operator", 5},
		{"admin", 1}, // Has wildcard
	}

	for _, tc := range testCases {
		role, ok := roles[tc.name]
		if !ok {
			t.Errorf("predefined role %s not found", tc.name)
			continue
		}
		if len(role.Permissions) < tc.expected {
			t.Errorf("role %s has %d permissions, expected at least %d",
				tc.name, len(role.Permissions), tc.expected)
		}
		if !role.IsBuiltIn {
			t.Errorf("predefined role %s should be built-in", tc.name)
		}
	}
}

// TestRBACManagerCreateRole tests creating custom roles
func TestRBACManagerCreateRole(t *testing.T) {
	rm := NewRBACManager("")

	role, err := rm.CreateRole("custom-role", "Custom role for testing", []Permission{
		PermVectorQuery,
		PermVectorInsert,
	})
	if err != nil {
		t.Fatalf("failed to create role: %v", err)
	}

	if role.Name != "custom-role" {
		t.Errorf("expected name 'custom-role', got %s", role.Name)
	}
	if role.IsBuiltIn {
		t.Error("custom role should not be built-in")
	}
	if len(role.Permissions) != 2 {
		t.Errorf("expected 2 permissions, got %d", len(role.Permissions))
	}
}

// TestRBACManagerCreateDuplicateRole tests rejecting duplicate role names
func TestRBACManagerCreateDuplicateRole(t *testing.T) {
	rm := NewRBACManager("")

	_, err := rm.CreateRole("test-role", "First role", []Permission{PermVectorQuery})
	if err != nil {
		t.Fatalf("failed to create first role: %v", err)
	}

	_, err = rm.CreateRole("test-role", "Duplicate role", []Permission{PermVectorInsert})
	if err == nil {
		t.Error("should reject duplicate role name")
	}
}

// TestRBACManagerUpdateRole tests updating custom roles
func TestRBACManagerUpdateRole(t *testing.T) {
	rm := NewRBACManager("")

	rm.CreateRole("test-role", "Test role", []Permission{PermVectorQuery})

	err := rm.UpdateRole("test-role", []Permission{PermVectorQuery, PermVectorInsert})
	if err != nil {
		t.Fatalf("failed to update role: %v", err)
	}

	role, _ := rm.GetRole("test-role")
	if len(role.Permissions) != 2 {
		t.Errorf("expected 2 permissions after update, got %d", len(role.Permissions))
	}
}

// TestRBACManagerUpdateBuiltInRole tests that built-in roles cannot be modified
func TestRBACManagerUpdateBuiltInRole(t *testing.T) {
	rm := NewRBACManager("")

	err := rm.UpdateRole("admin", []Permission{PermVectorQuery})
	if err == nil {
		t.Error("should not allow modifying built-in roles")
	}
}

// TestRBACManagerDeleteRole tests deleting custom roles
func TestRBACManagerDeleteRole(t *testing.T) {
	rm := NewRBACManager("")

	rm.CreateRole("test-role", "Test role", []Permission{PermVectorQuery})

	err := rm.DeleteRole("test-role")
	if err != nil {
		t.Fatalf("failed to delete role: %v", err)
	}

	_, ok := rm.GetRole("test-role")
	if ok {
		t.Error("role should be deleted")
	}
}

// TestRBACManagerDeleteBuiltInRole tests that built-in roles cannot be deleted
func TestRBACManagerDeleteBuiltInRole(t *testing.T) {
	rm := NewRBACManager("")

	err := rm.DeleteRole("admin")
	if err == nil {
		t.Error("should not allow deleting built-in roles")
	}
}

// TestRBACManagerAssignRole tests role assignment
func TestRBACManagerAssignRole(t *testing.T) {
	rm := NewRBACManager("")

	assignment, err := rm.AssignRole("user", "user-123", "viewer", "admin", nil, nil)
	if err != nil {
		t.Fatalf("failed to assign role: %v", err)
	}

	if assignment.SubjectID != "user-123" {
		t.Errorf("expected subject 'user-123', got %s", assignment.SubjectID)
	}
	if assignment.RoleName != "viewer" {
		t.Errorf("expected role 'viewer', got %s", assignment.RoleName)
	}
}

// TestRBACManagerAssignNonExistentRole tests assigning non-existent role
func TestRBACManagerAssignNonExistentRole(t *testing.T) {
	rm := NewRBACManager("")

	_, err := rm.AssignRole("user", "user-123", "non-existent-role", "admin", nil, nil)
	if err == nil {
		t.Error("should reject non-existent role")
	}
}

// TestRBACManagerAssignRoleWithExpiry tests role assignment with expiry
func TestRBACManagerAssignRoleWithExpiry(t *testing.T) {
	rm := NewRBACManager("")

	expiry := 24 * time.Hour
	assignment, err := rm.AssignRole("user", "user-123", "viewer", "admin", nil, &expiry)
	if err != nil {
		t.Fatalf("failed to assign role: %v", err)
	}

	if assignment.ExpiresAt == nil {
		t.Fatal("assignment should have expiry")
	}

	if time.Until(*assignment.ExpiresAt) > 25*time.Hour {
		t.Error("expiry too far in future")
	}
}

// TestRBACManagerAssignRoleWithScope tests role assignment with collection scope
func TestRBACManagerAssignRoleWithScope(t *testing.T) {
	rm := NewRBACManager("")

	scope := &Scope{Collections: []string{"coll1", "coll2"}}
	assignment, err := rm.AssignRole("user", "user-123", "editor", "admin", scope, nil)
	if err != nil {
		t.Fatalf("failed to assign role: %v", err)
	}

	if assignment.Scope == nil {
		t.Fatal("assignment should have scope")
	}
	if len(assignment.Scope.Collections) != 2 {
		t.Errorf("expected 2 collections, got %d", len(assignment.Scope.Collections))
	}
}

// TestRBACManagerRevokeRole tests role revocation
func TestRBACManagerRevokeRole(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "user-123", "viewer", "admin", nil, nil)

	err := rm.RevokeRole("user-123", "viewer", nil)
	if err != nil {
		t.Fatalf("failed to revoke role: %v", err)
	}

	assignments := rm.GetAssignments("user-123")
	if len(assignments) != 0 {
		t.Error("should have no assignments after revocation")
	}
}

// TestRBACManagerHasPermission tests permission checking
func TestRBACManagerHasPermission(t *testing.T) {
	rm := NewRBACManager("")

	// Assign viewer role
	rm.AssignRole("user", "user-123", "viewer", "admin", nil, nil)

	// Viewer should have query permission
	if !rm.HasPermission("user-123", PermVectorQuery, "") {
		t.Error("viewer should have vector:query permission")
	}

	// Viewer should not have insert permission
	if rm.HasPermission("user-123", PermVectorInsert, "") {
		t.Error("viewer should not have vector:insert permission")
	}
}

// TestRBACManagerHasPermissionAdmin tests admin wildcard permission
func TestRBACManagerHasPermissionAdmin(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "admin-user", "admin", "system", nil, nil)

	// Admin should have any permission via wildcard
	testPerms := []Permission{
		PermVectorQuery,
		PermVectorInsert,
		PermCollectionCreate,
		PermAdminUsers,
		PermSystemShutdown,
	}

	for _, perm := range testPerms {
		if !rm.HasPermission("admin-user", perm, "") {
			t.Errorf("admin should have %s permission", perm)
		}
	}
}

// TestRBACManagerHasPermissionWildcard tests category wildcard permissions
func TestRBACManagerHasPermissionWildcard(t *testing.T) {
	rm := NewRBACManager("")

	// Operator has vector:* and collection:*
	rm.AssignRole("user", "operator-user", "operator", "admin", nil, nil)

	// Should have all vector permissions
	if !rm.HasPermission("operator-user", PermVectorInsert, "") {
		t.Error("operator should have vector:insert via wildcard")
	}
	if !rm.HasPermission("operator-user", PermVectorDelete, "") {
		t.Error("operator should have vector:delete via wildcard")
	}

	// Should have all collection permissions
	if !rm.HasPermission("operator-user", PermCollectionCreate, "") {
		t.Error("operator should have collection:create via wildcard")
	}
}

// TestRBACManagerHasPermissionWithScope tests scoped permission checking
func TestRBACManagerHasPermissionWithScope(t *testing.T) {
	rm := NewRBACManager("")

	// Assign editor with scope limited to specific collections
	scope := &Scope{Collections: []string{"allowed-collection"}}
	rm.AssignRole("user", "scoped-user", "editor", "admin", scope, nil)

	// Should have permission for allowed collection
	if !rm.HasPermission("scoped-user", PermVectorInsert, "allowed-collection") {
		t.Error("should have permission for allowed collection")
	}

	// Should not have permission for other collection
	if rm.HasPermission("scoped-user", PermVectorInsert, "other-collection") {
		t.Error("should not have permission for other collection")
	}
}

// TestRBACManagerHasAnyPermission tests any permission checking
func TestRBACManagerHasAnyPermission(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "viewer-user", "viewer", "admin", nil, nil)

	// Should have any of [query, insert]
	if !rm.HasAnyPermission("viewer-user", []Permission{PermVectorQuery, PermVectorInsert}, "") {
		t.Error("should have at least one permission")
	}

	// Should not have any of [insert, delete]
	if rm.HasAnyPermission("viewer-user", []Permission{PermVectorInsert, PermVectorDelete}, "") {
		t.Error("should not have any of these permissions")
	}
}

// TestRBACManagerHasAllPermissions tests all permissions checking
func TestRBACManagerHasAllPermissions(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "editor-user", "editor", "admin", nil, nil)

	// Should have all of [query, insert]
	if !rm.HasAllPermissions("editor-user", []Permission{PermVectorQuery, PermVectorInsert}, "") {
		t.Error("should have all permissions")
	}

	// Should not have all of [query, admin:users]
	if rm.HasAllPermissions("editor-user", []Permission{PermVectorQuery, PermAdminUsers}, "") {
		t.Error("should not have admin:users permission")
	}
}

// TestRBACManagerExpiredAssignment tests that expired assignments are ignored
func TestRBACManagerExpiredAssignment(t *testing.T) {
	rm := NewRBACManager("")

	// Create assignment that expires immediately
	expiry := 1 * time.Millisecond
	rm.AssignRole("user", "temp-user", "admin", "system", nil, &expiry)

	// Wait for expiry
	time.Sleep(10 * time.Millisecond)

	// Should not have permission after expiry
	if rm.HasPermission("temp-user", PermVectorQuery, "") {
		t.Error("should not have permission after assignment expiry")
	}

	// GetAssignments should filter expired
	assignments := rm.GetAssignments("temp-user")
	if len(assignments) != 0 {
		t.Error("expired assignments should be filtered")
	}
}

// TestRBACManagerListRoles tests listing all roles
func TestRBACManagerListRoles(t *testing.T) {
	rm := NewRBACManager("")

	rm.CreateRole("custom1", "Custom 1", []Permission{PermVectorQuery})
	rm.CreateRole("custom2", "Custom 2", []Permission{PermVectorInsert})

	roles := rm.ListRoles()

	// Should have 4 predefined + 2 custom
	if len(roles) < 6 {
		t.Errorf("expected at least 6 roles, got %d", len(roles))
	}
}

// TestRBACManagerPermissionCaching tests that permission cache works
func TestRBACManagerPermissionCaching(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "cached-user", "viewer", "admin", nil, nil)

	// First check builds cache
	rm.HasPermission("cached-user", PermVectorQuery, "")

	// Check cache was built
	rm.mu.RLock()
	_, cached := rm.permCache["cached-user"]
	rm.mu.RUnlock()

	if !cached {
		t.Error("permission set should be cached")
	}

	// Second check should use cache (we can't directly verify, but it shouldn't error)
	if !rm.HasPermission("cached-user", PermVectorQuery, "") {
		t.Error("cached permission check failed")
	}
}

// TestRBACManagerCacheInvalidation tests cache invalidation
func TestRBACManagerCacheInvalidation(t *testing.T) {
	rm := NewRBACManager("")

	rm.AssignRole("user", "user-123", "viewer", "admin", nil, nil)

	// Build cache
	rm.HasPermission("user-123", PermVectorQuery, "")

	// Assign new role (should invalidate cache)
	rm.AssignRole("user", "user-123", "editor", "admin", nil, nil)

	// Should now have insert permission
	if !rm.HasPermission("user-123", PermVectorInsert, "") {
		t.Error("should have insert permission after new role assignment")
	}
}

// TestRBACManagerPersistence tests persistence to file
func TestRBACManagerPersistence(t *testing.T) {
	tmpFile := t.TempDir() + "/rbac.json"

	// Create manager and add data
	rm1 := NewRBACManager(tmpFile)
	rm1.CreateRole("persistent-role", "Test", []Permission{PermVectorQuery})
	rm1.AssignRole("user", "persistent-user", "persistent-role", "admin", nil, nil)

	// Create new manager that loads from file
	rm2 := NewRBACManager(tmpFile)

	// Check role was loaded
	role, ok := rm2.GetRole("persistent-role")
	if !ok {
		t.Error("custom role should be loaded from file")
	}
	if role.Name != "persistent-role" {
		t.Error("loaded role has wrong name")
	}

	// Check assignment was loaded
	assignments := rm2.GetAssignments("persistent-user")
	if len(assignments) != 1 {
		t.Errorf("expected 1 assignment, got %d", len(assignments))
	}
}

// TestRBACManagerMultipleRolesPerUser tests user with multiple roles
func TestRBACManagerMultipleRolesPerUser(t *testing.T) {
	rm := NewRBACManager("")

	// Assign multiple roles to same user
	rm.AssignRole("user", "multi-role-user", "viewer", "admin", nil, nil)

	// Assign custom role with additional permission
	rm.CreateRole("backup-operator", "Backup ops", []Permission{PermAdminBackup})
	rm.AssignRole("user", "multi-role-user", "backup-operator", "admin", nil, nil)

	// Should have permissions from both roles
	if !rm.HasPermission("multi-role-user", PermVectorQuery, "") {
		t.Error("should have viewer permission")
	}
	if !rm.HasPermission("multi-role-user", PermAdminBackup, "") {
		t.Error("should have backup permission")
	}

	assignments := rm.GetAssignments("multi-role-user")
	if len(assignments) != 2 {
		t.Errorf("expected 2 assignments, got %d", len(assignments))
	}
}
