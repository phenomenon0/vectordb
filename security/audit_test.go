package security

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestAuditLogger(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "audit_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	config := AuditConfig{
		LogDir:        tmpDir,
		MaxFileSize:   1024 * 1024,
		MaxFiles:      5,
		MinSeverity:   AuditSeverityInfo,
		BufferSize:    10,
		FlushInterval: 100 * time.Millisecond,
		AsyncWrite:    true,
		RetentionDays: 30,
		OutputFormat:  "jsonl",
		NodeID:        "test-node-1",
	}

	logger, err := NewAuditLogger(config)
	if err != nil {
		t.Fatalf("failed to create audit logger: %v", err)
	}
	defer logger.Close()

	// Test logging various events
	t.Run("LogAuth", func(t *testing.T) {
		logger.LogAuth(AuditEventLogin, "user123", "192.168.1.1", AuditOutcomeSuccess, map[string]any{
			"method": "password",
		})
		logger.LogAuth(AuditEventLoginFailed, "user456", "10.0.0.1", AuditOutcomeFailure, map[string]any{
			"reason": "invalid_password",
		})
	})

	t.Run("LogRBAC", func(t *testing.T) {
		logger.LogRBAC(AuditEventRoleAssigned, "admin", "user123", "editor", AuditOutcomeSuccess, nil)
		logger.LogRBAC(AuditEventRoleRevoked, "admin", "user456", "viewer", AuditOutcomeSuccess, nil)
	})

	t.Run("LogVector", func(t *testing.T) {
		logger.LogVector(AuditEventVectorInsert, "user123", "products", 100, AuditOutcomeSuccess, 5*time.Millisecond)
		logger.LogVector(AuditEventVectorQuery, "user123", "products", 10, AuditOutcomeSuccess, 2*time.Millisecond)
	})

	t.Run("LogCollection", func(t *testing.T) {
		logger.LogCollection(AuditEventCollCreate, "admin", "products", AuditOutcomeSuccess, map[string]any{
			"dimension": 128,
			"metric":    "cosine",
		})
	})

	t.Run("LogPermissionDenied", func(t *testing.T) {
		logger.LogPermissionDenied("user789", "admin:*", "", "192.168.1.50")
	})

	t.Run("LogAdmin", func(t *testing.T) {
		logger.LogAdmin(AuditEventBackup, "admin", "created_backup", AuditOutcomeSuccess, map[string]any{
			"backup_id": "backup-001",
		})
	})

	t.Run("LogCluster", func(t *testing.T) {
		logger.LogCluster(AuditEventLeaderElect, "node-2", "elected_as_leader", map[string]any{
			"term": 5,
		})
	})

	t.Run("LogSystem", func(t *testing.T) {
		logger.LogSystem(AuditEventStartup, "server_started", AuditSeverityInfo, map[string]any{
			"version": "1.0.0",
		})
	})

	// Force flush
	logger.Flush()
	time.Sleep(200 * time.Millisecond) // Wait for async writes

	// Verify files were created
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("failed to read audit dir: %v", err)
	}

	if len(entries) == 0 {
		t.Fatal("no audit files created")
	}

	t.Logf("Created %d audit file(s)", len(entries))
}

func TestAuditQuerier(t *testing.T) {
	// Create temp directory and logger
	tmpDir, err := os.MkdirTemp("", "audit_query_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	config := DefaultAuditConfig()
	config.LogDir = tmpDir
	config.NodeID = "test-node"
	config.AsyncWrite = false // Sync for testing

	logger, err := NewAuditLogger(config)
	if err != nil {
		t.Fatalf("failed to create audit logger: %v", err)
	}

	// Log some events
	subjects := []string{"alice", "bob", "charlie"}
	for i := 0; i < 30; i++ {
		subject := subjects[i%3]
		logger.LogVector(AuditEventVectorInsert, subject, "collection1", i+1, AuditOutcomeSuccess, time.Duration(i)*time.Millisecond)
	}

	// Add some auth events
	logger.LogAuth(AuditEventLogin, "alice", "192.168.1.1", AuditOutcomeSuccess, nil)
	logger.LogAuth(AuditEventLoginFailed, "mallory", "10.0.0.99", AuditOutcomeFailure, nil)
	logger.LogPermissionDenied("mallory", "admin:*", "", "10.0.0.99")

	logger.Flush()
	logger.Close()

	// Create querier
	querier := NewAuditQuerier(tmpDir)

	t.Run("QueryAll", func(t *testing.T) {
		result, err := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     100,
		})
		if err != nil {
			t.Fatalf("query failed: %v", err)
		}
		if result.TotalCount != 33 {
			t.Errorf("expected 33 events, got %d", result.TotalCount)
		}
	})

	t.Run("QueryBySubject", func(t *testing.T) {
		result, err := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			SubjectID: "alice",
			Limit:     100,
		})
		if err != nil {
			t.Fatalf("query failed: %v", err)
		}
		// alice has 10 vector inserts + 1 login = 11
		if result.TotalCount != 11 {
			t.Errorf("expected 11 events for alice, got %d", result.TotalCount)
		}
	})

	t.Run("QueryByEventType", func(t *testing.T) {
		result, err := querier.Query(AuditQuery{
			StartTime:  time.Now().Add(-1 * time.Hour),
			EndTime:    time.Now().Add(1 * time.Hour),
			EventTypes: []AuditEventType{AuditEventLoginFailed, AuditEventPermDenied},
			Limit:      100,
		})
		if err != nil {
			t.Fatalf("query failed: %v", err)
		}
		if result.TotalCount != 2 {
			t.Errorf("expected 2 security events, got %d", result.TotalCount)
		}
	})

	t.Run("QueryWithPagination", func(t *testing.T) {
		result1, err := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     10,
			Offset:    0,
		})
		if err != nil {
			t.Fatalf("query failed: %v", err)
		}
		if len(result1.Events) != 10 {
			t.Errorf("expected 10 events, got %d", len(result1.Events))
		}
		if !result1.HasMore {
			t.Error("expected HasMore=true")
		}

		result2, err := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     10,
			Offset:    10,
		})
		if err != nil {
			t.Fatalf("query failed: %v", err)
		}
		if len(result2.Events) != 10 {
			t.Errorf("expected 10 events, got %d", len(result2.Events))
		}
	})
}

func TestAuditStats(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "audit_stats_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	config := DefaultAuditConfig()
	config.LogDir = tmpDir
	config.NodeID = "test-node"
	config.AsyncWrite = false

	logger, err := NewAuditLogger(config)
	if err != nil {
		t.Fatalf("failed to create audit logger: %v", err)
	}

	// Log varied events
	for i := 0; i < 20; i++ {
		logger.LogVector(AuditEventVectorInsert, "user1", "col1", 10, AuditOutcomeSuccess, time.Millisecond)
	}
	for i := 0; i < 10; i++ {
		logger.LogVector(AuditEventVectorQuery, "user2", "col1", 5, AuditOutcomeSuccess, time.Millisecond)
	}
	for i := 0; i < 5; i++ {
		logger.LogAuth(AuditEventLoginFailed, "attacker", "1.2.3.4", AuditOutcomeFailure, nil)
	}
	logger.LogPermissionDenied("attacker", "admin:*", "", "1.2.3.4")

	logger.Flush()
	logger.Close()

	querier := NewAuditQuerier(tmpDir)
	stats, err := querier.GetStats(time.Now().Add(-1*time.Hour), time.Now().Add(1*time.Hour))
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}

	if stats.TotalEvents != 36 {
		t.Errorf("expected 36 total events, got %d", stats.TotalEvents)
	}

	if stats.EventsByType[AuditEventVectorInsert] != 20 {
		t.Errorf("expected 20 vector inserts, got %d", stats.EventsByType[AuditEventVectorInsert])
	}

	if stats.SecurityAlerts != 6 { // 5 failed logins + 1 permission denied
		t.Errorf("expected 6 security alerts, got %d", stats.SecurityAlerts)
	}

	// Check top subjects
	if len(stats.TopSubjects) == 0 {
		t.Error("expected top subjects")
	} else {
		t.Logf("Top subject: %s with %d events", stats.TopSubjects[0].SubjectID, stats.TopSubjects[0].EventCount)
	}
}

func TestAuditFiltering(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "audit_filter_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	t.Run("ExcludeEvents", func(t *testing.T) {
		config := DefaultAuditConfig()
		config.LogDir = filepath.Join(tmpDir, "exclude")
		config.NodeID = "test-exclude"
		config.AsyncWrite = false
		config.ExcludeEvents = []AuditEventType{AuditEventVectorQuery}

		logger, _ := NewAuditLogger(config)

		// Log both types
		logger.LogVector(AuditEventVectorInsert, "user1", "col1", 10, AuditOutcomeSuccess, time.Millisecond)
		logger.LogVector(AuditEventVectorQuery, "user1", "col1", 5, AuditOutcomeSuccess, time.Millisecond) // Should be excluded
		logger.Flush()
		logger.Close()

		querier := NewAuditQuerier(filepath.Join(tmpDir, "exclude"))
		result, _ := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     100,
		})

		if result.TotalCount != 1 {
			t.Errorf("expected 1 event (query excluded), got %d", result.TotalCount)
		}
	})

	t.Run("IncludeOnlyEvents", func(t *testing.T) {
		config := DefaultAuditConfig()
		config.LogDir = filepath.Join(tmpDir, "include")
		config.NodeID = "test-include"
		config.AsyncWrite = false
		config.IncludeEvents = []AuditEventType{AuditEventLogin, AuditEventLoginFailed}

		logger, _ := NewAuditLogger(config)

		// Log various types
		logger.LogAuth(AuditEventLogin, "user1", "1.1.1.1", AuditOutcomeSuccess, nil)
		logger.LogVector(AuditEventVectorInsert, "user1", "col1", 10, AuditOutcomeSuccess, time.Millisecond) // Should be excluded
		logger.LogAuth(AuditEventLoginFailed, "user2", "2.2.2.2", AuditOutcomeFailure, nil)
		logger.Flush()
		logger.Close()

		querier := NewAuditQuerier(filepath.Join(tmpDir, "include"))
		result, _ := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     100,
		})

		if result.TotalCount != 2 {
			t.Errorf("expected 2 events (only auth), got %d", result.TotalCount)
		}
	})

	t.Run("MinSeverity", func(t *testing.T) {
		config := DefaultAuditConfig()
		config.LogDir = filepath.Join(tmpDir, "severity")
		config.NodeID = "test-severity"
		config.AsyncWrite = false
		config.MinSeverity = AuditSeverityWarning

		logger, _ := NewAuditLogger(config)

		// Log events of different severities
		logger.LogVector(AuditEventVectorInsert, "user1", "col1", 10, AuditOutcomeSuccess, time.Millisecond) // Info - excluded
		logger.LogAuth(AuditEventLoginFailed, "user2", "1.1.1.1", AuditOutcomeFailure, nil)                  // Warning - included
		logger.LogAdmin(AuditEventConfigChange, "admin", "changed_config", AuditOutcomeSuccess, nil)         // Critical - included
		logger.Flush()
		logger.Close()

		querier := NewAuditQuerier(filepath.Join(tmpDir, "severity"))
		result, _ := querier.Query(AuditQuery{
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now().Add(1 * time.Hour),
			Limit:     100,
		})

		if result.TotalCount != 2 {
			t.Errorf("expected 2 events (warning+critical), got %d", result.TotalCount)
		}
	})
}

func TestAuditLogRotation(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "audit_rotation_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	config := DefaultAuditConfig()
	config.LogDir = tmpDir
	config.NodeID = "test-rotation"
	config.AsyncWrite = false
	config.MaxFileSize = 500 // Very small for testing

	logger, err := NewAuditLogger(config)
	if err != nil {
		t.Fatalf("failed to create audit logger: %v", err)
	}

	// Log many events to trigger rotation
	for i := 0; i < 50; i++ {
		logger.LogVector(AuditEventVectorInsert, "user1", "collection1", i, AuditOutcomeSuccess, time.Millisecond)
	}

	logger.Flush()
	logger.Close()

	// Check multiple files were created
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("failed to read dir: %v", err)
	}

	if len(entries) < 2 {
		t.Errorf("expected multiple log files due to rotation, got %d", len(entries))
	}

	t.Logf("Created %d audit files after rotation", len(entries))
}

func BenchmarkAuditLogging(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "audit_bench_*")
	defer os.RemoveAll(tmpDir)

	config := DefaultAuditConfig()
	config.LogDir = tmpDir
	config.NodeID = "bench-node"
	config.BufferSize = 5000
	config.AsyncWrite = true

	logger, _ := NewAuditLogger(config)
	defer logger.Close()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			logger.LogVector(AuditEventVectorInsert, "user1", "collection1", 100, AuditOutcomeSuccess, time.Millisecond)
		}
	})
}
