package collection

import (
	"context"
	"errors"
	"fmt"
)

// Single-leader read replicas. A replica DurableStore refuses every local
// write and advances only by applying its leader's journal records in LSN
// order, which keeps its own journal a record-for-record copy of the leader's.
//
// Pair with StreamJournal/FollowJournal on the leader: the leader streams,
// the replica applies, and the replica's own AppliedLSN is the durable cursor
// for where to resume.

var (
	// ErrReplicaReadOnly is returned for any write attempted directly against
	// a read replica. Accepting one would fork the two histories at the same
	// LSN, which is exactly the split brain the legacy cluster spike exhibited.
	ErrReplicaReadOnly = errors.New("durable collection store is a read replica")

	// ErrReplicaNotConfigured is returned when ApplyReplicated is called on a
	// store that was never marked a replica. Injecting a foreign journal record
	// into a leader would renumber its own history.
	ErrReplicaNotConfigured = errors.New("durable collection store is not a read replica")

	// ErrReplicaOutOfOrder is returned when a record is not the exact successor
	// of what the replica has already applied. The caller must resume from
	// ReplicaCursor rather than skip ahead.
	ErrReplicaOutOfOrder = errors.New("replicated record is not the replica's next LSN")
)

// MakeReplica marks the store as a read replica of leaderID.
//
// It must be called on every open: the binding is configuration, not state.
// The store must be empty or already exactly aligned with that leader's
// history; MakeReplica cannot verify that on its own, so the first
// ApplyReplicated rejects a misaligned store rather than renumbering records.
func (s *DurableStore) MakeReplica(leaderID [16]byte) error {
	if leaderID == ([16]byte{}) {
		return errors.New("replica leader store ID cannot be zero")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	if s.replica && s.leaderID != leaderID {
		return fmt.Errorf("%w: already replicating %x, asked for %x", ErrJournalStoreMismatch, s.leaderID, leaderID)
	}
	s.replica = true
	s.leaderID = leaderID
	return nil
}

// IsReplica reports whether local writes are refused.
func (s *DurableStore) IsReplica() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.replica
}

// ReplicaCursor is the position to resume the leader's stream from. It is
// derived from AppliedLSN, which is durable, so a restarted replica resumes
// exactly where it stopped without re-applying anything.
func (s *DurableStore) ReplicaCursor() JournalCursor {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return JournalCursor{StoreID: s.leaderID, LSN: s.metadata.AppliedLSN}
}

// ApplyReplicated durably applies one record streamed from the leader.
//
// The record's payload is appended verbatim, so the replica's journal is a
// record-for-record copy and can be streamed to a further follower. Apply is
// mandatory once the append succeeds: a failure there latches the store fault,
// the same contract the local write path has.
func (s *DurableStore) ApplyReplicated(ctx context.Context, record JournalRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	if !s.replica {
		return ErrReplicaNotConfigured
	}
	// The journal assigns LSNs from its own counter, so that counter — not
	// AppliedLSN — decides whether this record can be stored under the number
	// the leader gave it. Anything else silently renumbers the leader's history
	// and produces a journal no further follower can resume against. This one
	// comparison rejects a re-delivered record, a skipped record, and a replica
	// whose journal and applied state have drifted apart.
	if assigned := s.journalLastLSN() + 1; assigned != record.LSN {
		return fmt.Errorf(
			"%w: next durable LSN is %d (applied %d), leader sent LSN %d",
			ErrReplicaOutOfOrder, assigned, s.metadata.AppliedLSN, record.LSN,
		)
	}
	mutation, err := decodeDurableMutation(record.Payload)
	if err != nil {
		return fmt.Errorf("decode replicated mutation at LSN %d: %w", record.LSN, err)
	}
	// Same preparation as crash recovery: the leader already assigned document
	// IDs, so this re-resolves the target collection and normalizes vectors
	// without minting anything new.
	if err := s.prepareReplayMutation(&mutation); err != nil {
		return fmt.Errorf("validate replicated mutation at LSN %d: %w", record.LSN, err)
	}
	if err := s.appendPayloadApplyLocked(ctx, record.Payload, mutation); err != nil {
		return err
	}
	// Collection and tenant counters are maintained by the local write entry
	// points, not by the shared commit path. Recount instead of duplicating
	// that bookkeeping; it is O(tenants) and only runs on the two collection
	// lifecycle mutations, never per document.
	switch mutation.typeName {
	case mutationCreateCollection, mutationDeleteCollection:
		s.activeTenants, s.collectionCount = s.tenants.resourceCounts()
	}
	return nil
}
