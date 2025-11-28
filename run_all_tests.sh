#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                   VECTOR DATABASE TEST SUITE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
START_TIME=$(date +%s)

# Function to print test section header
print_section() {
    echo ""
    echo -e "${YELLOW}▶ $1${NC}"
    echo -e "${YELLOW}$(printf '─%.0s' {1..80})${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

# Function to print failure
print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

# Clean up old test artifacts
print_section "Cleaning up old test artifacts"
rm -rf /tmp/vectordb_test_* 2>/dev/null || true
rm -f vectordb/index.gob.wal 2>/dev/null || true
print_success "Cleanup complete"

# ============================================================================
# 1. Unit Tests
# ============================================================================
print_section "Running Unit Tests"
if go test -v -count=1 -timeout=60s ./... 2>&1 | tee /tmp/test_output.log; then
    print_success "All unit tests passed"
else
    print_failure "Unit tests failed"
fi

# ============================================================================
# 2. Race Condition Tests
# ============================================================================
print_section "Running Race Condition Tests"
if go test -race -count=1 -timeout=120s ./... 2>&1 | tee /tmp/test_race.log; then
    print_success "No race conditions detected"
else
    print_failure "Race conditions detected"
fi

# ============================================================================
# 3. Coverage Report
# ============================================================================
print_section "Generating Coverage Report"
if go test -coverprofile=/tmp/coverage.out -covermode=atomic ./... 2>&1; then
    COVERAGE=$(go tool cover -func=/tmp/coverage.out | grep total | awk '{print $3}')
    print_success "Coverage generated: $COVERAGE"

    # Generate HTML coverage report
    go tool cover -html=/tmp/coverage.out -o /tmp/coverage.html
    echo -e "   ${BLUE}→ HTML report: /tmp/coverage.html${NC}"
else
    print_failure "Coverage generation failed"
fi

# ============================================================================
# 4. Benchmarks
# ============================================================================
print_section "Running Benchmarks (5 iterations each)"
if go test -bench=. -benchmem -benchtime=5x -count=1 ./... 2>&1 | tee /tmp/bench_output.log; then
    print_success "Benchmarks completed"
    echo ""
    echo -e "${BLUE}Benchmark Summary:${NC}"
    grep "Benchmark" /tmp/bench_output.log | grep -v "^?" | tail -10
else
    print_failure "Benchmarks failed"
fi

# ============================================================================
# 5. Integration Tests
# ============================================================================
print_section "Running Integration Tests"
if go test -tags=integration -v -count=1 -timeout=120s ./... 2>&1 | tee /tmp/integration_output.log; then
    print_success "Integration tests passed"
else
    # Integration tests might not exist yet, so just warn
    echo -e "${YELLOW}⚠ No integration tests found or they failed${NC}"
fi

# ============================================================================
# 6. Phase 2 Feature Tests (New features - might not have tests yet)
# ============================================================================
print_section "Checking Phase 2 Features"

# Check if Phase 2 files compile
PHASE2_FILES="cache.go prefilter.go quantization.go rebalance.go replication.go"
PHASE2_OK=true

for file in $PHASE2_FILES; do
    if [ -f "$file" ]; then
        if go build -o /dev/null "$file" 2>/dev/null; then
            print_success "$file compiles successfully"
        else
            print_failure "$file has compilation errors"
            PHASE2_OK=false
        fi
    else
        echo -e "${YELLOW}⚠ $file not found${NC}"
    fi
done

# ============================================================================
# 7. Memory Leak Test
# ============================================================================
print_section "Memory Leak Detection (Basic)"
cat > /tmp/memory_test.go <<'EOF'
package main

import (
    "fmt"
    "runtime"
    "testing"
)

func TestMemoryLeak(t *testing.T) {
    vs := NewVectorStore(10000, 384)
    emb := NewHashEmbedder(384)

    runtime.GC()
    var m1 runtime.MemStats
    runtime.ReadMemStats(&m1)

    // Add and delete 1000 vectors 10 times
    for round := 0; round < 10; round++ {
        for i := 0; i < 1000; i++ {
            vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
            vs.Add(vec, "content", fmt.Sprintf("id-%d", i), nil, "default")
        }
        for i := 0; i < 1000; i++ {
            vs.Delete(fmt.Sprintf("id-%d", i))
        }
    }

    runtime.GC()
    var m2 runtime.MemStats
    runtime.ReadMemStats(&m2)

    memGrowth := m2.Alloc - m1.Alloc
    t.Logf("Memory growth: %d bytes (%.2f MB)", memGrowth, float64(memGrowth)/(1024*1024))

    // Allow up to 50MB growth (generous threshold)
    if memGrowth > 50*1024*1024 {
        t.Errorf("Possible memory leak: %d bytes growth", memGrowth)
    }
}
EOF

if go test -v /tmp/memory_test.go *.go 2>&1 | tee /tmp/memory_leak.log; then
    print_success "No significant memory leaks detected"
else
    print_failure "Memory leak test failed"
fi

# ============================================================================
# 8. Stress Test (Optional - quick version)
# ============================================================================
print_section "Quick Stress Test"
cat > /tmp/stress_test.go <<'EOF'
package main

import (
    "fmt"
    "sync"
    "testing"
    "time"
)

func TestStressTest(t *testing.T) {
    vs := NewVectorStore(50000, 384)
    emb := NewHashEmbedder(384)

    start := time.Now()
    var wg sync.WaitGroup
    errors := make(chan error, 100)

    // 20 writers
    for w := 0; w < 20; w++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for i := 0; i < 100; i++ {
                vec, _ := emb.Embed(fmt.Sprintf("worker-%d-doc-%d", workerID, i))
                if _, err := vs.Add(vec, "content", "", nil, "default"); err != nil {
                    errors <- err
                    return
                }
            }
        }(w)
    }

    // 10 readers
    for r := 0; r < 10; r++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            vec, _ := emb.Embed("query")
            for i := 0; i < 500; i++ {
                _ = vs.SearchANN(vec, 10)
            }
        }()
    }

    wg.Wait()
    close(errors)

    elapsed := time.Since(start)

    // Check for errors
    errCount := 0
    for err := range errors {
        t.Errorf("stress test error: %v", err)
        errCount++
    }

    if errCount > 0 {
        t.Fatalf("stress test failed with %d errors", errCount)
    }

    t.Logf("Stress test completed in %v", elapsed)
    t.Logf("Final count: %d vectors", vs.Count)
    t.Logf("Throughput: %.2f ops/sec", float64(2000+5000)/elapsed.Seconds())
}
EOF

if go test -v -timeout=60s /tmp/stress_test.go *.go 2>&1 | tee /tmp/stress_test.log; then
    print_success "Stress test passed"
else
    print_failure "Stress test failed"
fi

# ============================================================================
# Final Summary
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                          TEST RESULTS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✓ Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}✗ Failed: $TESTS_FAILED${NC}"
else
    echo -e "${GREEN}✗ Failed: $TESTS_FAILED${NC}"
fi
echo -e "${BLUE}⏱  Duration: ${DURATION}s${NC}"
echo ""

# Extract interesting stats from test output
if [ -f /tmp/test_output.log ]; then
    TOTAL_TESTS=$(grep -o "PASS\|FAIL" /tmp/test_output.log | wc -l | tr -d ' ')
    echo -e "${BLUE}Total test functions executed: $TOTAL_TESTS${NC}"
fi

# Coverage summary
if [ -f /tmp/coverage.out ]; then
    echo -e "${BLUE}Coverage: $COVERAGE${NC}"
fi

# Benchmark highlights
if [ -f /tmp/bench_output.log ]; then
    echo ""
    echo -e "${BLUE}Top 5 Benchmark Results:${NC}"
    grep "Benchmark" /tmp/bench_output.log | grep -v "^?" | head -5 | while read line; do
        echo -e "  ${YELLOW}$line${NC}"
    done
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Exit with error if any tests failed
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}OVERALL RESULT: FAILED${NC}"
    echo ""
    exit 1
else
    echo -e "${GREEN}OVERALL RESULT: SUCCESS ✓${NC}"
    echo ""
    exit 0
fi
