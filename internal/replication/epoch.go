package replication

import (
	"errors"
	"fmt"
	"os"
	"strings"
)

// epochSuffix, like markerSuffix, deliberately does not begin with a dot --
// see marker.go's comment on markerSuffix for why a dotted name would make a
// directory holding only a stale sidecar look occupied.
const epochSuffix = "-epoch"

func epochPath(basePath string) string { return basePath + epochSuffix }

// Epoch fences a leader's history. Number increases only at promotion; a
// follower that has adopted epoch N will not accept a stream from a leader
// claiming an epoch below N -- that leader was demoted. StartLSN is the
// journal position promotion happened at, so a follower can tell whether it
// already has records from the history the new epoch replaces: at or before
// StartLSN, it can adopt the new epoch and keep going; past it, its own
// records were never part of the new leader's history and it must resync.
type Epoch struct {
	Number   uint64
	StartLSN uint64
}

// WriteEpoch persists the epoch a store is following, beside it.
//
// 0600, same reasoning as MarkReplica: this file decides whether a leader's
// stream is honored, so it is not world-writable.
func WriteEpoch(basePath string, e Epoch) error {
	if basePath == "" {
		return errors.New("epoch base path cannot be empty")
	}
	content := fmt.Sprintf("%d %d\n", e.Number, e.StartLSN)
	if err := os.WriteFile(epochPath(basePath), []byte(content), 0o600); err != nil {
		return fmt.Errorf("write epoch sidecar: %w", err)
	}
	return nil
}

// ReadEpoch reports the epoch a store is following.
//
// A directory with no sidecar is Epoch{} -- a store that has never seen a
// promotion is at epoch 0 by definition. A sidecar that exists but cannot be
// read or parsed is an error, never a zero: the caller's next move is to
// decide whether to honor a leader's stream, and guessing epoch 0 from a
// corrupt file is how a demoted leader gets followed again. Same stance as
// ReplicaLeaderID.
func ReadEpoch(basePath string) (Epoch, error) {
	if basePath == "" {
		return Epoch{}, errors.New("epoch base path cannot be empty")
	}
	raw, err := os.ReadFile(epochPath(basePath))
	if errors.Is(err, os.ErrNotExist) {
		return Epoch{}, nil
	}
	if err != nil {
		return Epoch{}, fmt.Errorf("read epoch sidecar: %w", err)
	}
	var e Epoch
	if _, err := fmt.Sscanf(strings.TrimSpace(string(raw)), "%d %d", &e.Number, &e.StartLSN); err != nil {
		return Epoch{}, fmt.Errorf("epoch sidecar %q does not hold \"<number> <start_lsn>\"", epochPath(basePath))
	}
	return e, nil
}
