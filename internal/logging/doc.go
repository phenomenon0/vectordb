// Package logging provides structured logging for VectorDB using log/slog.
//
// By default, output is JSON (machine-parseable for production log aggregators).
// Set LOG_FORMAT=text for human-readable output during development.
//
// Tower layer: cross-cutting support; L0 through L5 all log through it.
package logging
