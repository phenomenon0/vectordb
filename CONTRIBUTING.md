# Contributing to VectorDB

Thank you for your interest in contributing to VectorDB! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Go 1.22 or later
- golangci-lint (for linting)
- Make (optional, but recommended)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/Agent-GO.git
cd Agent-GO

# Install dependencies
go mod download

# Run tests to verify setup
make test-short
```

## Development Workflow

### Running Tests

```bash
# Quick tests (recommended during development)
make test-short

# Full test suite
make test

# Tests with race detector
make test-race

# VectorDB-specific tests
make test-vectordb

# Tests with coverage report
make test-coverage
```

### Building

```bash
# Build all packages
make build

# Build VectorDB binary
make build-vectordb
```

### Linting

```bash
# Run linter
make lint

# Format code
make fmt
```

## Code Guidelines

### Code Style

- Follow standard Go conventions and idioms
- Use `gofmt` for formatting (or `make fmt`)
- Keep functions focused and reasonably sized
- Add comments for exported functions and types

### Error Handling

- Always handle errors explicitly
- Don't ignore errors with `_ =` unless absolutely necessary and documented
- Use `fmt.Errorf` with `%w` for error wrapping
- Prefer returning errors over panicking

### Testing

- Write tests for new functionality
- Use table-driven tests where appropriate
- Add `testing.Short()` skips for expensive tests:
  ```go
  func TestExpensiveOperation(t *testing.T) {
      if testing.Short() {
          t.Skip("skipping in short mode")
      }
      // ... test code
  }
  ```
- Run tests with race detector before submitting PRs

### Concurrency

- Document thread-safety guarantees in comments
- Use `sync.Mutex` or `sync.RWMutex` for shared state
- Avoid lock yielding patterns (unlock/lock in sequence)
- Prefer channels for communication between goroutines

## Pull Request Process

1. **Fork and Branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes**: Implement your changes following the guidelines above

3. **Test**: Ensure all tests pass
   ```bash
   make test-short
   make test-race
   make lint
   ```

4. **Commit**: Write clear commit messages
   ```
   feat(vectordb): add support for binary vectors
   
   - Implement binary vector type
   - Add Hamming distance metric
   - Update index interface
   ```

5. **Push and PR**: Push your branch and create a pull request

6. **Review**: Address any feedback from reviewers

## Commit Message Format

We follow conventional commits:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

Examples:
```
feat(index): add DiskANN support
fix(hnsw): resolve race condition in Add()
docs: update API reference
test(vectordb): add integration tests for hybrid search
```

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Go version, OS, VectorDB version
6. **Logs**: Relevant log output or error messages

## Feature Requests

For feature requests:

1. Check existing issues to avoid duplicates
2. Describe the use case and motivation
3. Propose a solution if you have one
4. Be open to discussion and alternatives

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions

## Questions?

If you have questions:

1. Check existing documentation in `/vectordb/QUICKSTART.md`
2. Search existing issues
3. Open a new issue with the "question" label

Thank you for contributing!
