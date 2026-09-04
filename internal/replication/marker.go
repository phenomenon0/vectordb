package replication

import (
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
)

// A replica directory has to say so on disk.
//
// DurableStore.replica is in-memory and deliberately not persisted: which
// leader a store follows is configuration, and putting it in the snapshot
// would tie it to a durable format version. But the flag is what makes a store
// refuse local writes, so without any on-disk trace a plain `deepdata serve`
// pointed at a replica directory opens it as an ordinary store and accepts
// writes -- forking the two histories at the same LSN, which is the exact
// split brain the replica guard exists to prevent.
//
// So the configuration is written beside the store, by the one process that
// knows it: the follower. The data directory is the evidence, and an operator
// cannot get it wrong by forgetting a flag.

// markerSuffix deliberately does NOT begin with a dot. Every durable store
// artifact is basePath+"."+suffix, and both BootstrapReplica's emptiness scan
// and storeArtifactsExist treat any such name as "a store lives here". A
// dotted marker would make a directory holding only a stale marker look
// occupied and send a re-sync down the resume path instead of bootstrapping.
const markerSuffix = "-replica"

func markerPath(basePath string) string { return basePath + markerSuffix }

// MarkReplica records which leader the store at basePath replicates.
//
// It must be called only after the store is known to be a replica of leaderID,
// and never before BootstrapReplica has run: the marker sits outside the
// artifact prefix precisely so it cannot be mistaken for store state, but
// writing it early would still leave a marker pointing at a store that does
// not exist.
func MarkReplica(basePath string, leaderID [16]byte) error {
	if basePath == "" {
		return errors.New("replica base path cannot be empty")
	}
	if leaderID == ([16]byte{}) {
		return errors.New("replica leader store ID cannot be zero")
	}
	// 0600: the leader ID is not a secret, but this file decides whether a
	// directory is servable read-only, so it is not world-writable.
	if err := os.WriteFile(markerPath(basePath), []byte(hex.EncodeToString(leaderID[:])+"\n"), 0o600); err != nil {
		return fmt.Errorf("write replica marker: %w", err)
	}
	return nil
}

// ReplicaLeaderID reports the leader a directory replicates, and whether the
// directory is a replica at all.
//
// A marker that exists but cannot be read or parsed is an error, never a
// false: the caller's next move is to serve the directory, and guessing
// "ordinary store" from an unreadable marker is how a replica ends up
// accepting writes.
func ReplicaLeaderID(basePath string) ([16]byte, bool, error) {
	var id [16]byte
	if basePath == "" {
		return id, false, errors.New("replica base path cannot be empty")
	}
	raw, err := os.ReadFile(markerPath(basePath))
	if errors.Is(err, os.ErrNotExist) {
		return id, false, nil
	}
	if err != nil {
		return id, false, fmt.Errorf("read replica marker: %w", err)
	}
	decoded, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil || len(decoded) != len(id) {
		return id, false, fmt.Errorf("replica marker %q does not hold a 32-character hex store ID", markerPath(basePath))
	}
	copy(id[:], decoded)
	return id, true, nil
}
