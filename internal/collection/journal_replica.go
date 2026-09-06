package collection

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
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
	// that bookkeeping; it is O(tenants) and only runs on the lifecycle
	// mutations, never per document.
	switch mutation.typeName {
	case mutationCreateCollection, mutationDeleteCollection,
		mutationCreateTenant, mutationUpdateTenant, mutationDeleteTenant:
		s.activeTenants, s.collectionCount = s.tenants.resourceCounts()
	}
	return nil
}

// BootstrapReplica materializes a read replica at basePath from a leader
// snapshot stream (WriteSnapshot) and binds it to leaderID.
//
// This is what lets a replica join a leader that is already running. A leader
// deletes every journal artifact a checkpoint has covered, so a store starting
// from LSN 0 gets ErrJournalGap from StreamJournal forever; it has to be handed
// state instead. The snapshot carries the leader's AppliedLSN in its header, so
// the opened store's journal counter is seeded there and the first record the
// leader streams -- AppliedLSN+1 -- is exactly the one ApplyReplicated accepts.
//
// basePath is a file PREFIX, not a directory: the store's artifacts are
// basePath+".snapshot", basePath+".journal" and so on. Its parent directory
// must already exist, the same contract OpenDurableStore has.
//
// The store comes back already read-only. There is deliberately no window in
// which the caller holds an unmarked store: a single local write in that window
// would consume the LSN the leader's next record needs and wedge the replica
// for good.
func BootstrapReplica(basePath, storagePath string, r io.Reader, leaderID [16]byte) (*DurableStore, error) {
	if leaderID == ([16]byte{}) {
		return nil, errors.New("replica leader store ID cannot be zero")
	}
	if basePath == "" {
		return nil, errors.New("durable collection store base path cannot be empty")
	}
	if r == nil {
		return nil, errors.New("replica bootstrap snapshot reader cannot be nil")
	}
	// Same normalization openDurableStore does, so the scan below and the open
	// below are talking about the same name.
	basePath = filepath.Clean(basePath)

	// Refuse a path that already holds a store BEFORE writing a byte: the copy
	// lands on basePath+".snapshot" and would otherwise destroy an existing
	// store's newest generation with no way back. Every artifact is
	// basePath+"."+suffix (.snapshot .manager .tenants .initialized .journal
	// .journal.frozen .usage.json .lock), so matching on the dot covers all of
	// them with no suffix list to keep in sync. The dot is load-bearing, not
	// cosmetic: a bare prefix also matches a SIBLING store whose name merely
	// extends this one, so bootstrapping "shard1" would be refused because
	// "shard10.initialized" sits beside it. Unlike filepath.Glob this has no
	// metacharacter hazard -- a '[' in the path would make Glob match nothing
	// and silently proceed to overwrite.
	entries, err := os.ReadDir(filepath.Dir(basePath))
	if err != nil {
		return nil, fmt.Errorf("inspect replica bootstrap directory: %w", err)
	}
	prefix := filepath.Base(basePath) + "."
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), prefix) {
			return nil, fmt.Errorf("replica bootstrap target %q is not empty: found %q", basePath, entry.Name())
		}
	}

	// Temp file, chmod 0600, fsync, rename, sync the parent. The loader reads
	// the snapshot twice and rejects one whose bytes or mode changed between
	// passes, so a torn transfer must never appear under the real name.
	if err := writeCollectionFileAtomicStream(collectionSnapshotPath(basePath), 0o600, func(w io.Writer) error {
		_, copyErr := io.Copy(w, r)
		return copyErr
	}); err != nil {
		return nil, fmt.Errorf("receive leader snapshot: %w", err)
	}

	// The normal open path does everything that matters for free: it verifies
	// the v2 checksum, adopts the leader's StoreID and AppliedLSN from the
	// header, mints the .initialized marker from that same ID, and seeds the
	// journal with lastLSN = AppliedLSN. Writing any of those by hand would be
	// a second author of a format that already has one.
	//
	// A failed open deliberately leaves the copied snapshot behind: removing
	// only it could leave a marker with no snapshot, which is permanently
	// unopenable, and the emptiness scan above makes the debris loud.
	store, err := OpenDurableStore(basePath, storagePath)
	if err != nil {
		return nil, fmt.Errorf("open bootstrapped replica: %w", err)
	}
	// Trust boundary. MakeReplica cannot tell where the state came from, and a
	// snapshot that never landed leaves OpenDurableStore minting a fresh random
	// StoreID -- the replica then journals the leader's records under its own
	// identity and fails much later, much more confusingly.
	if id := store.Metadata().StoreID; id != leaderID {
		// Abort, not Close: Close would checkpoint a store we are rejecting.
		return nil, errors.Join(
			fmt.Errorf("%w: bootstrap snapshot is from store %x, expected leader %x", ErrJournalStoreMismatch, id, leaderID),
			store.Abort(),
		)
	}
	if err := store.MakeReplica(leaderID); err != nil {
		return nil, errors.Join(err, store.Abort())
	}
	return store, nil
}

// ponytail: the replica adopts the leader's StoreID, because it arrives inside
// the checksummed snapshot header and rewriting it would mean a second snapshot
// encoder. Ceiling: after a future promotion two stores share one StoreID with
// divergent LSNs, so ErrJournalStoreMismatch cannot catch a follower repointed
// between the old and new leader. Upgrade trigger is the day Promote() lands --
// it must mint a new StoreID or carry an epoch.
