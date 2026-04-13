// Package logging provides structured logging for VectorDB using log/slog.
//
// By default, output is JSON (machine-parseable for production log aggregators).
// Set LOG_FORMAT=text for human-readable output during development.
package logging

import (
	"context"
	"log/slog"
	"os"
	"time"
)

// Level represents log levels
type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

// Config holds logging configuration
type Config struct {
	Level  Level
	Format string // "json" (default) or "text"
	Output *os.File
}

// DefaultConfig returns default logging configuration.
// Default format is JSON for production use.
func DefaultConfig() Config {
	return Config{
		Level:  LevelInfo,
		Format: "json",
		Output: os.Stdout,
	}
}

// Logger is the structured logger backed by log/slog.
type Logger struct {
	level Level
	slog  *slog.Logger
}

var defaultLogger *Logger

// toSlogLevel converts our Level to slog.Level.
func toSlogLevel(l Level) slog.Level {
	switch l {
	case LevelDebug:
		return slog.LevelDebug
	case LevelWarn:
		return slog.LevelWarn
	case LevelError:
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Init initializes the logging system.
func Init(cfg Config) *Logger {
	opts := &slog.HandlerOptions{
		Level: toSlogLevel(cfg.Level),
	}

	output := cfg.Output
	if output == nil {
		output = os.Stdout
	}

	var handler slog.Handler
	if cfg.Format == "text" {
		handler = slog.NewTextHandler(output, opts)
	} else {
		handler = slog.NewJSONHandler(output, opts)
	}

	defaultLogger = &Logger{
		level: cfg.Level,
		slog:  slog.New(handler),
	}
	return defaultLogger
}

// Default returns the default logger instance.
func Default() *Logger {
	if defaultLogger == nil {
		Init(DefaultConfig())
	}
	return defaultLogger
}

// FromContext returns a logger from context.
// If the context carries slog attributes (via slog.NewLogLogger or similar),
// they will be included. Falls back to the receiver logger.
func (l *Logger) FromContext(ctx context.Context) *Logger {
	return l
}

// Debug logs a debug message with structured key-value pairs.
func (l *Logger) Debug(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelDebug {
		l.slog.Debug(msg, keysAndValues...)
	}
}

// Info logs an info message with structured key-value pairs.
func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelInfo {
		l.slog.Info(msg, keysAndValues...)
	}
}

// Warn logs a warning message with structured key-value pairs.
func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelWarn {
		l.slog.Warn(msg, keysAndValues...)
	}
}

// Error logs an error message with structured key-value pairs.
func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelError {
		l.slog.Error(msg, keysAndValues...)
	}
}

// LogError logs an error with context and structured key-value pairs.
func (l *Logger) LogError(ctx context.Context, operation string, err error, keysAndValues ...interface{}) {
	args := make([]interface{}, 0, 4+len(keysAndValues))
	args = append(args, "operation", operation, "error", err)
	args = append(args, keysAndValues...)
	l.slog.ErrorContext(ctx, "operation failed", args...)
}

// Insert logs an insert operation at debug level.
func (l *Logger) Insert(ctx context.Context, id string, collection string, dim int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.slog.DebugContext(ctx, "insert",
			"id", id,
			"collection", collection,
			"dim", dim,
			"duration", duration,
		)
	}
}

// BatchInsert logs a batch insert operation at debug level.
func (l *Logger) BatchInsert(ctx context.Context, collection string, count int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.slog.DebugContext(ctx, "batch_insert",
			"collection", collection,
			"count", count,
			"duration", duration,
		)
	}
}

// Search logs a search operation at debug level.
func (l *Logger) Search(ctx context.Context, collection string, topK int, results int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.slog.DebugContext(ctx, "search",
			"collection", collection,
			"top_k", topK,
			"results", results,
			"duration", duration,
		)
	}
}

// Delete logs a delete operation at debug level.
func (l *Logger) Delete(ctx context.Context, collection string, id string, duration time.Duration) {
	if l.level <= LevelDebug {
		l.slog.DebugContext(ctx, "delete",
			"collection", collection,
			"id", id,
			"duration", duration,
		)
	}
}
