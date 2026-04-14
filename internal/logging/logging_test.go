package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// helper: init a logger that writes to a buffer and return the buffer.
func initBuf(level Level, format string) (*Logger, *bytes.Buffer) {
	var buf bytes.Buffer
	// We need an *os.File for the Config, but slog handlers accept io.Writer.
	// To test with a buffer we create a temp file, but that's heavy.
	// Instead, we'll use a pipe trick: write to a temp file and read back.
	// Actually, the Config takes *os.File. Let's use a pipe via os.Pipe.
	r, w, err := os.Pipe()
	if err != nil {
		panic(err)
	}

	cfg := Config{Level: level, Format: format, Output: w}
	logger := Init(cfg)

	// We'll need to close w and read from r after logging.
	// Store them for the caller. We use a goroutine to drain r into buf.
	go func() {
		tmp := make([]byte, 4096)
		for {
			n, err := r.Read(tmp)
			if n > 0 {
				buf.Write(tmp[:n])
			}
			if err != nil {
				break
			}
		}
	}()

	// Stash the write end so we can close it later.
	// We'll return the logger and use a wrapper.
	logger.slog.Info("__init__") // force a write so pipe goroutine starts
	_ = w // keep reference

	// Return a wrapper that closes w and waits a beat for the pipe to drain.
	// Actually, let's simplify: return w reference via a closure the caller uses.
	return logger, &buf
}

// flushPipe is a no-op since we switched to a simpler approach below.

// simpler approach: use a temp file.
func initToFile(level Level, format string) (*Logger, string) {
	f, err := os.CreateTemp("", "logtest-*.log")
	if err != nil {
		panic(err)
	}
	name := f.Name()
	cfg := Config{Level: level, Format: format, Output: f}
	Init(cfg)
	l := Default()
	_ = f // keep open
	return l, name
}

func readAndClean(path string) string {
	// Give a tiny window for file flush
	time.Sleep(10 * time.Millisecond)
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	os.Remove(path)
	return string(data)
}

func TestJSONFormat(t *testing.T) {
	logger, path := initToFile(LevelInfo, "json")
	logger.Info("hello world", "key1", "val1", "num", 42)
	// Close the file to flush
	logger.slog.Handler()
	output := readAndClean(path)

	if output == "" {
		t.Fatal("expected JSON output, got empty")
	}

	// Each line should be valid JSON
	lines := strings.Split(strings.TrimSpace(output), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Errorf("line is not valid JSON: %s\nerror: %v", line, err)
		}
	}
}

func TestJSONContainsFields(t *testing.T) {
	logger, path := initToFile(LevelInfo, "json")
	logger.Info("test msg", "mykey", "myval")
	output := readAndClean(path)

	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) == 0 {
		t.Fatal("no output")
	}
	last := lines[len(lines)-1]

	var m map[string]interface{}
	if err := json.Unmarshal([]byte(last), &m); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	// slog JSON output has: time, level, msg, plus our keys
	if m["msg"] != "test msg" {
		t.Errorf("msg = %v, want 'test msg'", m["msg"])
	}
	if m["level"] != "INFO" {
		t.Errorf("level = %v, want INFO", m["level"])
	}
	if m["mykey"] != "myval" {
		t.Errorf("mykey = %v, want 'myval'", m["mykey"])
	}
	if _, ok := m["time"]; !ok {
		t.Error("missing 'time' field in JSON output")
	}
}

func TestTextFormat(t *testing.T) {
	logger, path := initToFile(LevelInfo, "text")
	logger.Info("text output", "foo", "bar")
	output := readAndClean(path)

	if output == "" {
		t.Fatal("expected text output, got empty")
	}

	// Text format should NOT be valid JSON
	lines := strings.Split(strings.TrimSpace(output), "\n")
	last := lines[len(lines)-1]
	var m map[string]interface{}
	if json.Unmarshal([]byte(last), &m) == nil {
		t.Error("text format produced valid JSON — expected human-readable text")
	}

	// Should contain our key=value
	if !strings.Contains(last, "foo=bar") {
		t.Errorf("text output missing foo=bar: %s", last)
	}
}

func TestLevelFiltering(t *testing.T) {
	tests := []struct {
		name      string
		cfgLevel  Level
		logMethod string
		wantEmpty bool
	}{
		{"debug_at_info_level", LevelInfo, "debug", true},
		{"info_at_info_level", LevelInfo, "info", false},
		{"warn_at_info_level", LevelInfo, "warn", false},
		{"error_at_info_level", LevelInfo, "error", false},
		{"debug_at_debug_level", LevelDebug, "debug", false},
		{"info_at_warn_level", LevelWarn, "info", true},
		{"warn_at_warn_level", LevelWarn, "warn", false},
		{"info_at_error_level", LevelError, "info", true},
		{"warn_at_error_level", LevelError, "warn", true},
		{"error_at_error_level", LevelError, "error", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, path := initToFile(tt.cfgLevel, "json")
			switch tt.logMethod {
			case "debug":
				logger.Debug("test")
			case "info":
				logger.Info("test")
			case "warn":
				logger.Warn("test")
			case "error":
				logger.Error("test")
			}
			output := readAndClean(path)

			hasContent := strings.Contains(output, `"msg":"test"`)
			if tt.wantEmpty && hasContent {
				t.Errorf("expected message to be filtered, but got: %s", output)
			}
			if !tt.wantEmpty && !hasContent {
				t.Errorf("expected message in output, but it was filtered. output: %s", output)
			}
		})
	}
}

func TestLogError(t *testing.T) {
	logger, path := initToFile(LevelInfo, "json")
	testErr := errors.New("something broke")
	logger.LogError(context.Background(), "db_query", testErr, "table", "users")
	output := readAndClean(path)

	var m map[string]interface{}
	lines := strings.Split(strings.TrimSpace(output), "\n")
	last := lines[len(lines)-1]
	if err := json.Unmarshal([]byte(last), &m); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, last)
	}

	if m["level"] != "ERROR" {
		t.Errorf("level = %v, want ERROR", m["level"])
	}
	if m["operation"] != "db_query" {
		t.Errorf("operation = %v, want db_query", m["operation"])
	}
	if m["error"] != "something broke" {
		t.Errorf("error = %v, want 'something broke'", m["error"])
	}
	if m["table"] != "users" {
		t.Errorf("table = %v, want 'users'", m["table"])
	}
}

func TestLogErrorIncludesRequestID(t *testing.T) {
	logger, path := initToFile(LevelInfo, "json")
	testErr := errors.New("db timeout")

	// Context with a request ID should emit request_id in the log
	ctx := context.WithValue(context.Background(), RequestIDKey, "abc123def456")
	logger.LogError(ctx, "query", testErr, "collection", "docs")
	output := readAndClean(path)

	var m map[string]interface{}
	lines := strings.Split(strings.TrimSpace(output), "\n")
	last := lines[len(lines)-1]
	if err := json.Unmarshal([]byte(last), &m); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, last)
	}
	if m["request_id"] != "abc123def456" {
		t.Errorf("request_id = %v, want 'abc123def456'", m["request_id"])
	}
	if m["collection"] != "docs" {
		t.Errorf("collection = %v, want 'docs'", m["collection"])
	}
}

func TestLogErrorOmitsRequestIDWhenAbsent(t *testing.T) {
	logger, path := initToFile(LevelInfo, "json")
	testErr := errors.New("db timeout")

	// Context without request ID should NOT emit request_id field
	logger.LogError(context.Background(), "query", testErr)
	output := readAndClean(path)

	var m map[string]interface{}
	lines := strings.Split(strings.TrimSpace(output), "\n")
	last := lines[len(lines)-1]
	if err := json.Unmarshal([]byte(last), &m); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, last)
	}
	if _, exists := m["request_id"]; exists {
		t.Errorf("request_id should not be present when context has no request ID, got %v", m["request_id"])
	}
}

func TestOperationLogs(t *testing.T) {
	logger, path := initToFile(LevelDebug, "json")
	ctx := context.Background()

	logger.Insert(ctx, "vec-1", "docs", 768, 5*time.Millisecond)
	logger.BatchInsert(ctx, "docs", 100, 50*time.Millisecond)
	logger.Search(ctx, "docs", 10, 8, 3*time.Millisecond)
	logger.Delete(ctx, "docs", "vec-2", 1*time.Millisecond)

	output := readAndClean(path)
	lines := strings.Split(strings.TrimSpace(output), "\n")

	ops := []string{"insert", "batch_insert", "search", "delete"}
	for i, op := range ops {
		if i >= len(lines) {
			t.Errorf("missing log line for %s", op)
			continue
		}
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(lines[i]), &m); err != nil {
			t.Errorf("line %d invalid JSON: %v", i, err)
			continue
		}
		if m["msg"] != op {
			t.Errorf("line %d msg = %v, want %s", i, m["msg"], op)
		}
		if m["collection"] != "docs" {
			t.Errorf("line %d collection = %v, want docs", i, m["collection"])
		}
	}
}

func TestDefaultLoggerLazyInit(t *testing.T) {
	// Reset the default logger
	defaultLogger = nil
	l := Default()
	if l == nil {
		t.Fatal("Default() returned nil")
	}
	// Should be able to log without panic
	l.Info("lazy init works")
}

func TestToSlogLevel(t *testing.T) {
	tests := []struct {
		in   Level
		want string
	}{
		{LevelDebug, "DEBUG"},
		{LevelInfo, "INFO"},
		{LevelWarn, "WARN"},
		{LevelError, "ERROR"},
	}
	for _, tt := range tests {
		got := toSlogLevel(tt.in)
		if got.String() != tt.want {
			t.Errorf("toSlogLevel(%d) = %s, want %s", tt.in, got.String(), tt.want)
		}
	}
}
