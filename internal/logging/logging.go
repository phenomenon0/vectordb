// Package logging provides structured logging for VectorDB
package logging

import (
	"context"
	"log"
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
	Format string
	Output *os.File
}

// DefaultConfig returns default logging configuration
func DefaultConfig() Config {
	return Config{
		Level:  LevelInfo,
		Output: os.Stdout,
	}
}

// Logger is the structured logger
type Logger struct {
	level  Level
	logger *log.Logger
}

var defaultLogger *Logger

// Init initializes the logging system
func Init(cfg Config) *Logger {
	defaultLogger = &Logger{
		level:  cfg.Level,
		logger: log.New(cfg.Output, "[vectordb] ", log.LstdFlags),
	}
	return defaultLogger
}

// Default returns the default logger instance
func Default() *Logger {
	if defaultLogger == nil {
		Init(DefaultConfig())
	}
	return defaultLogger
}

// FromContext returns a logger from context (returns default for now)
func (l *Logger) FromContext(ctx context.Context) *Logger {
	return l
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelDebug {
		l.logger.Printf("[DEBUG] %s %v", msg, keysAndValues)
	}
}

// Info logs an info message
func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelInfo {
		l.logger.Printf("[INFO] %s %v", msg, keysAndValues)
	}
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelWarn {
		l.logger.Printf("[WARN] %s %v", msg, keysAndValues)
	}
}

// Error logs an error message
func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
	if l.level <= LevelError {
		l.logger.Printf("[ERROR] %s %v", msg, keysAndValues)
	}
}

// LogError logs an error with context
func (l *Logger) LogError(ctx context.Context, operation string, err error, keysAndValues ...interface{}) {
	l.logger.Printf("[ERROR] %s: %v %v", operation, err, keysAndValues)
}

// Insert logs an insert operation
func (l *Logger) Insert(ctx context.Context, id string, collection string, dim int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.logger.Printf("[DEBUG] insert id=%s collection=%s dim=%d duration=%v", id, collection, dim, duration)
	}
}

// BatchInsert logs a batch insert operation
func (l *Logger) BatchInsert(ctx context.Context, collection string, count int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.logger.Printf("[DEBUG] batch_insert collection=%s count=%d duration=%v", collection, count, duration)
	}
}

// Search logs a search operation
func (l *Logger) Search(ctx context.Context, collection string, topK int, results int, duration time.Duration) {
	if l.level <= LevelDebug {
		l.logger.Printf("[DEBUG] search collection=%s top_k=%d results=%d duration=%v", collection, topK, results, duration)
	}
}

// Delete logs a delete operation
func (l *Logger) Delete(ctx context.Context, collection string, id string, duration time.Duration) {
	if l.level <= LevelDebug {
		l.logger.Printf("[DEBUG] delete collection=%s id=%s duration=%v", collection, id, duration)
	}
}
