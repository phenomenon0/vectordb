package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ===========================================================================================
// AUDIT LOGGING
// Comprehensive audit trail for security, compliance, and debugging
// Records who did what, when, from where, and the outcome
// ===========================================================================================

// AuditEventType categorizes audit events
type AuditEventType string

const (
	// Authentication events
	AuditEventLogin        AuditEventType = "auth.login"
	AuditEventLogout       AuditEventType = "auth.logout"
	AuditEventLoginFailed  AuditEventType = "auth.login_failed"
	AuditEventTokenRefresh AuditEventType = "auth.token_refresh"
	AuditEventTokenRevoked AuditEventType = "auth.token_revoked"

	// Authorization events
	AuditEventPermDenied  AuditEventType = "authz.permission_denied"
	AuditEventPermGranted AuditEventType = "authz.permission_granted"

	// RBAC events
	AuditEventRoleCreated  AuditEventType = "rbac.role_created"
	AuditEventRoleDeleted  AuditEventType = "rbac.role_deleted"
	AuditEventRoleModified AuditEventType = "rbac.role_modified"
	AuditEventRoleAssigned AuditEventType = "rbac.role_assigned"
	AuditEventRoleRevoked  AuditEventType = "rbac.role_revoked"

	// Vector operations
	AuditEventVectorInsert AuditEventType = "vector.insert"
	AuditEventVectorQuery  AuditEventType = "vector.query"
	AuditEventVectorDelete AuditEventType = "vector.delete"
	AuditEventVectorUpdate AuditEventType = "vector.update"

	// Collection operations
	AuditEventCollCreate AuditEventType = "collection.create"
	AuditEventCollDelete AuditEventType = "collection.delete"
	AuditEventCollModify AuditEventType = "collection.modify"

	// Admin operations
	AuditEventBackup       AuditEventType = "admin.backup"
	AuditEventRestore      AuditEventType = "admin.restore"
	AuditEventConfigChange AuditEventType = "admin.config_change"
	AuditEventUserCreate   AuditEventType = "admin.user_create"
	AuditEventUserDelete   AuditEventType = "admin.user_delete"
	AuditEventUserModify   AuditEventType = "admin.user_modify"

	// System events
	AuditEventStartup  AuditEventType = "system.startup"
	AuditEventShutdown AuditEventType = "system.shutdown"
	AuditEventError    AuditEventType = "system.error"

	// Cluster events
	AuditEventLeaderElect AuditEventType = "cluster.leader_elected"
	AuditEventNodeJoin    AuditEventType = "cluster.node_join"
	AuditEventNodeLeave   AuditEventType = "cluster.node_leave"
	AuditEventReplication AuditEventType = "cluster.replication"
)

// AuditOutcome represents the result of an audited action
type AuditOutcome string

const (
	AuditOutcomeSuccess AuditOutcome = "success"
	AuditOutcomeFailure AuditOutcome = "failure"
	AuditOutcomeDenied  AuditOutcome = "denied"
	AuditOutcomeError   AuditOutcome = "error"
)

// AuditSeverity indicates the importance level of an audit event
type AuditSeverity string

const (
	AuditSeverityInfo     AuditSeverity = "info"
	AuditSeverityWarning  AuditSeverity = "warning"
	AuditSeverityCritical AuditSeverity = "critical"
)

// AuditEvent represents a single audit log entry
type AuditEvent struct {
	ID           string         `json:"id"`
	Timestamp    time.Time      `json:"timestamp"`
	EventType    AuditEventType `json:"event_type"`
	Severity     AuditSeverity  `json:"severity"`
	Outcome      AuditOutcome   `json:"outcome"`
	SubjectID    string         `json:"subject_id"`              // Who performed the action
	SubjectType  string         `json:"subject_type,omitempty"`  // user, service, system
	ResourceType string         `json:"resource_type,omitempty"` // collection, vector, role
	ResourceID   string         `json:"resource_id,omitempty"`   // Specific resource affected
	Action       string         `json:"action"`                  // Human-readable action
	Details      map[string]any `json:"details,omitempty"`       // Additional context
	SourceIP     string         `json:"source_ip,omitempty"`     // Client IP address
	UserAgent    string         `json:"user_agent,omitempty"`    // Client user agent
	RequestID    string         `json:"request_id,omitempty"`    // Correlation ID
	Duration     time.Duration  `json:"duration_ns,omitempty"`   // Operation duration
	ErrorMessage string         `json:"error_message,omitempty"` // Error details if failed
	NodeID       string         `json:"node_id,omitempty"`       // Cluster node ID
}

// AuditConfig configures the audit logger
type AuditConfig struct {
	// Storage settings
	LogDir           string        `json:"log_dir"`
	MaxFileSize      int64         `json:"max_file_size_bytes"` // Max size before rotation
	MaxFiles         int           `json:"max_files"`           // Max number of files to keep
	RotationInterval time.Duration `json:"rotation_interval"`   // Time-based rotation

	// Filtering settings
	MinSeverity   AuditSeverity    `json:"min_severity"`
	IncludeEvents []AuditEventType `json:"include_events,omitempty"` // If set, only these events
	ExcludeEvents []AuditEventType `json:"exclude_events,omitempty"` // Events to exclude

	// Performance settings
	BufferSize    int           `json:"buffer_size"` // In-memory buffer before flush
	FlushInterval time.Duration `json:"flush_interval"`
	AsyncWrite    bool          `json:"async_write"`

	// Retention settings
	RetentionDays int `json:"retention_days"`

	// Output settings
	OutputFormat    string `json:"output_format"` // json, jsonl
	CompressRotated bool   `json:"compress_rotated"`

	// Node identification
	NodeID string `json:"node_id"`
}

// DefaultAuditConfig returns sensible defaults
func DefaultAuditConfig() AuditConfig {
	return AuditConfig{
		LogDir:           "audit_logs",
		MaxFileSize:      100 * 1024 * 1024, // 100MB
		MaxFiles:         30,
		RotationInterval: 24 * time.Hour,
		MinSeverity:      AuditSeverityInfo,
		BufferSize:       1000,
		FlushInterval:    5 * time.Second,
		AsyncWrite:       true,
		RetentionDays:    90,
		OutputFormat:     "jsonl",
		CompressRotated:  true,
	}
}

// AuditLogger provides thread-safe audit logging with buffering and rotation
type AuditLogger struct {
	mu          sync.RWMutex
	config      AuditConfig
	currentFile *os.File
	currentSize int64
	currentDate string // For date-based rotation
	buffer      []*AuditEvent
	eventCount  uint64
	stopCh      chan struct{}
	flushCh     chan struct{}
	wg          sync.WaitGroup
	includeMap  map[AuditEventType]bool
	excludeMap  map[AuditEventType]bool
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger(config AuditConfig) (*AuditLogger, error) {
	// Create log directory
	if err := os.MkdirAll(config.LogDir, 0750); err != nil {
		return nil, fmt.Errorf("failed to create audit log directory: %w", err)
	}

	al := &AuditLogger{
		config:     config,
		buffer:     make([]*AuditEvent, 0, config.BufferSize),
		stopCh:     make(chan struct{}),
		flushCh:    make(chan struct{}, 1),
		includeMap: make(map[AuditEventType]bool),
		excludeMap: make(map[AuditEventType]bool),
	}

	// Build filter maps
	for _, e := range config.IncludeEvents {
		al.includeMap[e] = true
	}
	for _, e := range config.ExcludeEvents {
		al.excludeMap[e] = true
	}

	// Open initial log file
	if err := al.openNewFile(); err != nil {
		return nil, err
	}

	// Start background flusher if async
	if config.AsyncWrite {
		al.wg.Add(1)
		go al.flushLoop()
	}

	return al, nil
}

// Log records an audit event
func (al *AuditLogger) Log(event *AuditEvent) {
	// Apply filters
	if !al.shouldLog(event) {
		return
	}

	// Generate ID if not set
	if event.ID == "" {
		al.mu.Lock()
		al.eventCount++
		event.ID = fmt.Sprintf("%s-%d-%d", al.config.NodeID, event.Timestamp.UnixNano(), al.eventCount)
		al.mu.Unlock()
	}

	// Set timestamp if not set
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}

	// Set node ID
	if event.NodeID == "" {
		event.NodeID = al.config.NodeID
	}

	if al.config.AsyncWrite {
		al.mu.Lock()
		al.buffer = append(al.buffer, event)
		shouldFlush := len(al.buffer) >= al.config.BufferSize
		al.mu.Unlock()

		if shouldFlush {
			select {
			case al.flushCh <- struct{}{}:
			default:
			}
		}
	} else {
		al.mu.Lock()
		al.writeEvent(event)
		al.mu.Unlock()
	}
}

// LogAuth logs an authentication event
func (al *AuditLogger) LogAuth(eventType AuditEventType, subjectID, sourceIP string, outcome AuditOutcome, details map[string]any) {
	al.Log(&AuditEvent{
		Timestamp:   time.Now().UTC(),
		EventType:   eventType,
		Severity:    al.authSeverity(eventType, outcome),
		Outcome:     outcome,
		SubjectID:   subjectID,
		SubjectType: "user",
		Action:      string(eventType),
		SourceIP:    sourceIP,
		Details:     details,
	})
}

// LogRBAC logs an RBAC event
func (al *AuditLogger) LogRBAC(eventType AuditEventType, subjectID, targetSubject, roleName string, outcome AuditOutcome, details map[string]any) {
	if details == nil {
		details = make(map[string]any)
	}
	details["target_subject"] = targetSubject
	details["role_name"] = roleName

	al.Log(&AuditEvent{
		Timestamp:    time.Now().UTC(),
		EventType:    eventType,
		Severity:     AuditSeverityWarning,
		Outcome:      outcome,
		SubjectID:    subjectID,
		SubjectType:  "user",
		ResourceType: "role",
		ResourceID:   roleName,
		Action:       string(eventType),
		Details:      details,
	})
}

// LogVector logs a vector operation
func (al *AuditLogger) LogVector(eventType AuditEventType, subjectID, collection string, vectorCount int, outcome AuditOutcome, duration time.Duration) {
	al.Log(&AuditEvent{
		Timestamp:    time.Now().UTC(),
		EventType:    eventType,
		Severity:     AuditSeverityInfo,
		Outcome:      outcome,
		SubjectID:    subjectID,
		SubjectType:  "user",
		ResourceType: "collection",
		ResourceID:   collection,
		Action:       string(eventType),
		Duration:     duration,
		Details: map[string]any{
			"vector_count": vectorCount,
		},
	})
}

// LogCollection logs a collection operation
func (al *AuditLogger) LogCollection(eventType AuditEventType, subjectID, collectionName string, outcome AuditOutcome, details map[string]any) {
	al.Log(&AuditEvent{
		Timestamp:    time.Now().UTC(),
		EventType:    eventType,
		Severity:     AuditSeverityWarning,
		Outcome:      outcome,
		SubjectID:    subjectID,
		SubjectType:  "user",
		ResourceType: "collection",
		ResourceID:   collectionName,
		Action:       string(eventType),
		Details:      details,
	})
}

// LogPermissionDenied logs when a permission check fails
func (al *AuditLogger) LogPermissionDenied(subjectID, permission, collection, sourceIP string) {
	al.Log(&AuditEvent{
		Timestamp:    time.Now().UTC(),
		EventType:    AuditEventPermDenied,
		Severity:     AuditSeverityWarning,
		Outcome:      AuditOutcomeDenied,
		SubjectID:    subjectID,
		SubjectType:  "user",
		ResourceType: "permission",
		ResourceID:   permission,
		Action:       "permission_check",
		SourceIP:     sourceIP,
		Details: map[string]any{
			"permission_requested": permission,
			"collection":           collection,
		},
	})
}

// LogAdmin logs an admin operation
func (al *AuditLogger) LogAdmin(eventType AuditEventType, subjectID, action string, outcome AuditOutcome, details map[string]any) {
	al.Log(&AuditEvent{
		Timestamp:   time.Now().UTC(),
		EventType:   eventType,
		Severity:    AuditSeverityCritical,
		Outcome:     outcome,
		SubjectID:   subjectID,
		SubjectType: "admin",
		Action:      action,
		Details:     details,
	})
}

// LogCluster logs a cluster event
func (al *AuditLogger) LogCluster(eventType AuditEventType, nodeID, action string, details map[string]any) {
	al.Log(&AuditEvent{
		Timestamp:    time.Now().UTC(),
		EventType:    eventType,
		Severity:     AuditSeverityInfo,
		Outcome:      AuditOutcomeSuccess,
		SubjectID:    "system",
		SubjectType:  "system",
		ResourceType: "cluster",
		ResourceID:   nodeID,
		Action:       action,
		Details:      details,
	})
}

// LogSystem logs a system event
func (al *AuditLogger) LogSystem(eventType AuditEventType, action string, severity AuditSeverity, details map[string]any) {
	al.Log(&AuditEvent{
		Timestamp:   time.Now().UTC(),
		EventType:   eventType,
		Severity:    severity,
		Outcome:     AuditOutcomeSuccess,
		SubjectID:   "system",
		SubjectType: "system",
		Action:      action,
		Details:     details,
	})
}

// shouldLog checks if an event should be logged based on filters
func (al *AuditLogger) shouldLog(event *AuditEvent) bool {
	// Check exclude list first
	if al.excludeMap[event.EventType] {
		return false
	}

	// If include list is set, event must be in it
	if len(al.includeMap) > 0 && !al.includeMap[event.EventType] {
		return false
	}

	// Check severity
	return al.severityLevel(event.Severity) >= al.severityLevel(al.config.MinSeverity)
}

// severityLevel converts severity to numeric level
func (al *AuditLogger) severityLevel(s AuditSeverity) int {
	switch s {
	case AuditSeverityInfo:
		return 0
	case AuditSeverityWarning:
		return 1
	case AuditSeverityCritical:
		return 2
	default:
		return 0
	}
}

// authSeverity determines severity for auth events
func (al *AuditLogger) authSeverity(eventType AuditEventType, outcome AuditOutcome) AuditSeverity {
	if outcome == AuditOutcomeFailure || outcome == AuditOutcomeDenied {
		return AuditSeverityWarning
	}
	if eventType == AuditEventTokenRevoked {
		return AuditSeverityWarning
	}
	return AuditSeverityInfo
}

// flushLoop periodically flushes the buffer
func (al *AuditLogger) flushLoop() {
	defer al.wg.Done()

	ticker := time.NewTicker(al.config.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-al.stopCh:
			al.flush()
			return
		case <-ticker.C:
			al.flush()
		case <-al.flushCh:
			al.flush()
		}
	}
}

// flush writes buffered events to disk
func (al *AuditLogger) flush() {
	al.mu.Lock()
	defer al.mu.Unlock()

	if len(al.buffer) == 0 {
		return
	}

	// Check for rotation before writing
	al.checkRotation()

	for _, event := range al.buffer {
		al.writeEvent(event)
	}

	al.buffer = al.buffer[:0]
}

// writeEvent writes a single event to the current file
func (al *AuditLogger) writeEvent(event *AuditEvent) {
	if al.currentFile == nil {
		return
	}

	data, err := json.Marshal(event)
	if err != nil {
		return
	}

	data = append(data, '\n')
	n, err := al.currentFile.Write(data)
	if err != nil {
		return
	}

	al.currentSize += int64(n)
}

// openNewFile opens a new audit log file
func (al *AuditLogger) openNewFile() error {
	now := time.Now().UTC()
	al.currentDate = now.Format("2006-01-02")

	filename := fmt.Sprintf("audit-%s-%s.log", al.config.NodeID, now.Format("2006-01-02-150405"))
	filepath := filepath.Join(al.config.LogDir, filename)

	f, err := os.OpenFile(filepath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0640)
	if err != nil {
		return fmt.Errorf("failed to open audit file: %w", err)
	}

	// Close old file if exists
	if al.currentFile != nil {
		al.currentFile.Close()
	}

	al.currentFile = f
	al.currentSize = 0

	// Get actual file size if appending
	info, err := f.Stat()
	if err == nil {
		al.currentSize = info.Size()
	}

	return nil
}

// checkRotation checks if log rotation is needed
func (al *AuditLogger) checkRotation() {
	needsRotation := false

	// Size-based rotation
	if al.config.MaxFileSize > 0 && al.currentSize >= al.config.MaxFileSize {
		needsRotation = true
	}

	// Date-based rotation
	currentDate := time.Now().UTC().Format("2006-01-02")
	if currentDate != al.currentDate {
		needsRotation = true
	}

	if needsRotation {
		al.openNewFile()
		go al.cleanupOldFiles()
	}
}

// cleanupOldFiles removes old audit files beyond retention
func (al *AuditLogger) cleanupOldFiles() {
	entries, err := os.ReadDir(al.config.LogDir)
	if err != nil {
		return
	}

	cutoff := time.Now().AddDate(0, 0, -al.config.RetentionDays)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		if info.ModTime().Before(cutoff) {
			os.Remove(filepath.Join(al.config.LogDir, entry.Name()))
		}
	}
}

// Close flushes remaining events and closes the audit logger
func (al *AuditLogger) Close() error {
	close(al.stopCh)
	al.wg.Wait()

	al.mu.Lock()
	defer al.mu.Unlock()

	if al.currentFile != nil {
		return al.currentFile.Close()
	}
	return nil
}

// Flush forces an immediate flush of buffered events
func (al *AuditLogger) Flush() {
	al.flush()
}

// ===========================================================================================
// AUDIT QUERY INTERFACE
// Search and retrieve audit events for compliance and investigation
// ===========================================================================================

// AuditQuery defines search criteria for audit events
type AuditQuery struct {
	StartTime    time.Time        `json:"start_time"`
	EndTime      time.Time        `json:"end_time"`
	SubjectID    string           `json:"subject_id,omitempty"`
	EventTypes   []AuditEventType `json:"event_types,omitempty"`
	Severity     AuditSeverity    `json:"severity,omitempty"` // Minimum severity
	Outcome      AuditOutcome     `json:"outcome,omitempty"`
	ResourceType string           `json:"resource_type,omitempty"`
	ResourceID   string           `json:"resource_id,omitempty"`
	SourceIP     string           `json:"source_ip,omitempty"`
	Limit        int              `json:"limit"`
	Offset       int              `json:"offset"`
}

// AuditQueryResult holds query results
type AuditQueryResult struct {
	Events     []*AuditEvent `json:"events"`
	TotalCount int           `json:"total_count"`
	HasMore    bool          `json:"has_more"`
}

// AuditQuerier provides read access to audit logs
type AuditQuerier struct {
	logDir string
}

// NewAuditQuerier creates a querier for audit logs
func NewAuditQuerier(logDir string) *AuditQuerier {
	return &AuditQuerier{logDir: logDir}
}

// Query searches audit logs based on criteria
func (aq *AuditQuerier) Query(query AuditQuery) (*AuditQueryResult, error) {
	if query.Limit <= 0 {
		query.Limit = 100
	}
	if query.Limit > 10000 {
		query.Limit = 10000
	}

	// Find relevant log files
	files, err := aq.findFiles(query.StartTime, query.EndTime)
	if err != nil {
		return nil, err
	}

	var allEvents []*AuditEvent

	// Read and filter events from each file
	for _, file := range files {
		events, err := aq.readAndFilter(file, query)
		if err != nil {
			continue // Skip corrupt files
		}
		allEvents = append(allEvents, events...)
	}

	// Sort by timestamp descending (newest first)
	sort.Slice(allEvents, func(i, j int) bool {
		return allEvents[i].Timestamp.After(allEvents[j].Timestamp)
	})

	totalCount := len(allEvents)

	// Apply pagination
	start := query.Offset
	if start > len(allEvents) {
		start = len(allEvents)
	}
	end := start + query.Limit
	if end > len(allEvents) {
		end = len(allEvents)
	}

	return &AuditQueryResult{
		Events:     allEvents[start:end],
		TotalCount: totalCount,
		HasMore:    end < totalCount,
	}, nil
}

// findFiles finds log files within the time range
func (aq *AuditQuerier) findFiles(start, end time.Time) ([]string, error) {
	entries, err := os.ReadDir(aq.logDir)
	if err != nil {
		return nil, err
	}

	var files []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		// Include files modified within or after the time range
		// This is conservative - we may read some irrelevant files
		if info.ModTime().Before(start) {
			continue
		}

		files = append(files, filepath.Join(aq.logDir, entry.Name()))
	}

	return files, nil
}

// readAndFilter reads a log file and filters events
func (aq *AuditQuerier) readAndFilter(filepath string, query AuditQuery) ([]*AuditEvent, error) {
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var events []*AuditEvent
	decoder := json.NewDecoder(f)

	for {
		var event AuditEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			continue // Skip malformed lines
		}

		if aq.matchesQuery(&event, query) {
			events = append(events, &event)
		}
	}

	return events, nil
}

// matchesQuery checks if an event matches query criteria
func (aq *AuditQuerier) matchesQuery(event *AuditEvent, query AuditQuery) bool {
	// Time range check
	if !query.StartTime.IsZero() && event.Timestamp.Before(query.StartTime) {
		return false
	}
	if !query.EndTime.IsZero() && event.Timestamp.After(query.EndTime) {
		return false
	}

	// Subject filter
	if query.SubjectID != "" && event.SubjectID != query.SubjectID {
		return false
	}

	// Event type filter
	if len(query.EventTypes) > 0 {
		found := false
		for _, t := range query.EventTypes {
			if event.EventType == t {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Outcome filter
	if query.Outcome != "" && event.Outcome != query.Outcome {
		return false
	}

	// Resource filters
	if query.ResourceType != "" && event.ResourceType != query.ResourceType {
		return false
	}
	if query.ResourceID != "" && event.ResourceID != query.ResourceID {
		return false
	}

	// Source IP filter
	if query.SourceIP != "" && event.SourceIP != query.SourceIP {
		return false
	}

	return true
}

// ===========================================================================================
// HTTP HANDLERS FOR AUDIT LOG ACCESS
// ===========================================================================================

// AuditHandlers provides HTTP handlers for audit log access
type AuditHandlers struct {
	logger  *AuditLogger
	querier *AuditQuerier
	rbac    *RBACManager // For permission checks
}

// NewAuditHandlers creates HTTP handlers for audit access
func NewAuditHandlers(logger *AuditLogger, querier *AuditQuerier, rbac *RBACManager) *AuditHandlers {
	return &AuditHandlers{
		logger:  logger,
		querier: querier,
		rbac:    rbac,
	}
}

// HandleQuery handles POST /admin/audit/query
func (h *AuditHandlers) HandleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse query from body
	var query AuditQuery
	if err := json.NewDecoder(r.Body).Decode(&query); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Execute query
	result, err := h.querier.Query(query)
	if err != nil {
		http.Error(w, fmt.Sprintf("query failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleGetRecent handles GET /admin/audit/recent?limit=N
func (h *AuditHandlers) HandleGetRecent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	limit := 100
	if l := r.URL.Query().Get("limit"); l != "" {
		fmt.Sscanf(l, "%d", &limit)
	}
	if limit > 1000 {
		limit = 1000
	}

	query := AuditQuery{
		StartTime: time.Now().Add(-24 * time.Hour),
		EndTime:   time.Now(),
		Limit:     limit,
	}

	result, err := h.querier.Query(query)
	if err != nil {
		http.Error(w, fmt.Sprintf("query failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleGetBySubject handles GET /admin/audit/subject/{subject_id}
func (h *AuditHandlers) HandleGetBySubject(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	subjectID := r.URL.Query().Get("subject_id")
	if subjectID == "" {
		http.Error(w, "subject_id required", http.StatusBadRequest)
		return
	}

	days := 7
	if d := r.URL.Query().Get("days"); d != "" {
		fmt.Sscanf(d, "%d", &days)
	}
	if days > 90 {
		days = 90
	}

	query := AuditQuery{
		StartTime: time.Now().AddDate(0, 0, -days),
		EndTime:   time.Now(),
		SubjectID: subjectID,
		Limit:     500,
	}

	result, err := h.querier.Query(query)
	if err != nil {
		http.Error(w, fmt.Sprintf("query failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleGetSecurityEvents handles GET /admin/audit/security
func (h *AuditHandlers) HandleGetSecurityEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	hours := 24
	if hrs := r.URL.Query().Get("hours"); hrs != "" {
		fmt.Sscanf(hrs, "%d", &hours)
	}
	if hours > 168 { // Max 7 days
		hours = 168
	}

	// Query for security-relevant events
	query := AuditQuery{
		StartTime: time.Now().Add(-time.Duration(hours) * time.Hour),
		EndTime:   time.Now(),
		EventTypes: []AuditEventType{
			AuditEventLoginFailed,
			AuditEventPermDenied,
			AuditEventTokenRevoked,
			AuditEventRoleAssigned,
			AuditEventRoleRevoked,
			AuditEventConfigChange,
		},
		Limit: 1000,
	}

	result, err := h.querier.Query(query)
	if err != nil {
		http.Error(w, fmt.Sprintf("query failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleFlush handles POST /admin/audit/flush
func (h *AuditHandlers) HandleFlush(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	h.logger.Flush()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "flushed"})
}

// ===========================================================================================
// AUDIT STATISTICS AND METRICS
// ===========================================================================================

// AuditStats provides statistics about audit events
type AuditStats struct {
	TotalEvents      int64                    `json:"total_events"`
	EventsByType     map[AuditEventType]int64 `json:"events_by_type"`
	EventsBySeverity map[AuditSeverity]int64  `json:"events_by_severity"`
	EventsByOutcome  map[AuditOutcome]int64   `json:"events_by_outcome"`
	TopSubjects      []SubjectStat            `json:"top_subjects"`
	SecurityAlerts   int64                    `json:"security_alerts"`
	TimeRange        TimeRange                `json:"time_range"`
}

// SubjectStat tracks activity per subject
type SubjectStat struct {
	SubjectID  string    `json:"subject_id"`
	EventCount int64     `json:"event_count"`
	LastActive time.Time `json:"last_active"`
}

// TimeRange represents a time window
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// GetStats computes statistics for a time range
func (aq *AuditQuerier) GetStats(start, end time.Time) (*AuditStats, error) {
	query := AuditQuery{
		StartTime: start,
		EndTime:   end,
		Limit:     100000, // Large limit for stats
	}

	result, err := aq.Query(query)
	if err != nil {
		return nil, err
	}

	stats := &AuditStats{
		EventsByType:     make(map[AuditEventType]int64),
		EventsBySeverity: make(map[AuditSeverity]int64),
		EventsByOutcome:  make(map[AuditOutcome]int64),
		TimeRange:        TimeRange{Start: start, End: end},
	}

	subjectCounts := make(map[string]*SubjectStat)

	for _, event := range result.Events {
		stats.TotalEvents++
		stats.EventsByType[event.EventType]++
		stats.EventsBySeverity[event.Severity]++
		stats.EventsByOutcome[event.Outcome]++

		// Track subjects
		if s, ok := subjectCounts[event.SubjectID]; ok {
			s.EventCount++
			if event.Timestamp.After(s.LastActive) {
				s.LastActive = event.Timestamp
			}
		} else {
			subjectCounts[event.SubjectID] = &SubjectStat{
				SubjectID:  event.SubjectID,
				EventCount: 1,
				LastActive: event.Timestamp,
			}
		}

		// Count security alerts
		if event.Severity == AuditSeverityCritical ||
			event.EventType == AuditEventLoginFailed ||
			event.EventType == AuditEventPermDenied {
			stats.SecurityAlerts++
		}
	}

	// Convert subject map to sorted slice
	for _, s := range subjectCounts {
		stats.TopSubjects = append(stats.TopSubjects, *s)
	}
	sort.Slice(stats.TopSubjects, func(i, j int) bool {
		return stats.TopSubjects[i].EventCount > stats.TopSubjects[j].EventCount
	})
	if len(stats.TopSubjects) > 10 {
		stats.TopSubjects = stats.TopSubjects[:10]
	}

	return stats, nil
}

// HandleGetStats handles GET /admin/audit/stats
func (h *AuditHandlers) HandleGetStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	days := 7
	if d := r.URL.Query().Get("days"); d != "" {
		fmt.Sscanf(d, "%d", &days)
	}
	if days > 90 {
		days = 90
	}

	stats, err := h.querier.GetStats(
		time.Now().AddDate(0, 0, -days),
		time.Now(),
	)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to compute stats: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}
