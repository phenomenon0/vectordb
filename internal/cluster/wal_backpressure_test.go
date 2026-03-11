package cluster

import (
	"testing"
	"time"
)

func TestWALStreamConfigurableBuffer(t *testing.T) {
	// Default config
	ws := NewWALStream()
	if ws.maxEntries != 10000 {
		t.Fatalf("expected default maxEntries 10000, got %d", ws.maxEntries)
	}

	// Custom large buffer
	ws2 := NewWALStreamWithConfig(WALStreamConfig{MaxEntries: 100000})
	if ws2.maxEntries != 100000 {
		t.Fatalf("expected maxEntries 100000, got %d", ws2.maxEntries)
	}

	// Fill and verify trimming at new threshold
	for i := 0; i < 100500; i++ {
		ws2.Append(WalEntry{Op: "insert", ID: "v"})
	}
	if len(ws2.entries) > 100000 {
		t.Fatalf("buffer exceeded limit: %d > 100000", len(ws2.entries))
	}
}

func TestFollowerAdaptivePollConfig(t *testing.T) {
	cfg := DefaultFollowerReplicatorConfig()

	if cfg.PollInterval != 1*time.Second {
		t.Fatalf("expected default PollInterval 1s, got %v", cfg.PollInterval)
	}
	if cfg.FastPollInterval != 100*time.Millisecond {
		t.Fatalf("expected default FastPollInterval 100ms, got %v", cfg.FastPollInterval)
	}
}

func TestWALStreamBatchRetrieval(t *testing.T) {
	ws := NewWALStreamWithConfig(WALStreamConfig{MaxEntries: 50000})

	// Add 20K entries
	for i := 0; i < 20000; i++ {
		ws.Append(WalEntry{Op: "insert", ID: "v"})
	}

	// Retrieve in batches
	batchSize := 1000
	var totalRetrieved int
	lastSeq := uint64(0)

	for {
		entries, err := ws.GetSince(lastSeq)
		if err != nil {
			t.Fatalf("GetSince(%d) failed: %v", lastSeq, err)
		}
		if len(entries) == 0 {
			break
		}

		batch := entries
		if len(batch) > batchSize {
			batch = batch[:batchSize]
		}
		totalRetrieved += len(batch)
		lastSeq = batch[len(batch)-1].Seq

		if totalRetrieved >= 20000 {
			break
		}
	}

	if totalRetrieved != 20000 {
		t.Fatalf("expected to retrieve 20000 entries, got %d", totalRetrieved)
	}
}
