# Contributing to DeepData

## Development Setup

### Prerequisites

- Go 1.24 or later
- golangci-lint (for linting)

### Getting Started

```bash
git clone https://github.com/phenomenon0/vectordb.git
cd vectordb

go mod download
go test -short ./...
```

## Development Workflow

### Running Tests

```bash
# Quick tests (recommended during development)
go test -short ./...

# Full test suite
go test ./...

# Tests with race detector
go test -race ./...

# DeepData server tests
go test ./cmd/deepdata/...

# Tests with coverage
go test -coverprofile=cover.out ./...
go tool cover -html=cover.out
```

### Building

```bash
# Build server
go build -o deepdata ./cmd/deepdata

# Build CLI
go build -o deepdata-cli ./cmd/cli
```

### Linting

```bash
golangci-lint run ./...
gofmt -w .
```

## Code Guidelines

### Code Style

- Follow standard Go conventions and idioms
- Use `gofmt` for formatting
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
   go test -short ./...
   go test -race ./...
   golangci-lint run ./...
   ```

4. **Commit**: Write clear commit messages
   ```
   feat(index): add support for binary vectors

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

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Go version, OS, DeepData version
6. **Logs**: Relevant log output or error messages

## Feature Requests

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

1. Check the [README](../README.md) and [docs/](./)
2. Search existing issues
3. Open a new issue with the "question" label
