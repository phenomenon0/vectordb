package collection

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

// This file is the leader half of single-leader replication: it lets a caller
// read a durable store's journal as an ordered LSN stream, either once
// (StreamJournal) or continuously (FollowJournal).
//
// Two properties come from the journal format rather than from this code, and
// they are the reason replication is built here rather than on the retired
// flat-array WAL:
//
//   - The LSN is durable. It is recovered from the snapshot metadata and the
//     surviving artifacts at open, so a leader that restarts resumes numbering
//     where it stopped. A follower's cursor can never be silently ahead of the
//     leader's counter.
//   - Every frame carries the store UUID and a checksum. A follower pointed at
//     a different store, or at a store restored from a backup, is rejected by
//     the scanner instead of quietly interleaving foreign records.

// JournalRecord is one durably committed mutation, in LSN order.
//
// Payload aliases the reader's bounded buffer and is only valid until the
// visitor returns. A visitor that retains it must copy it.
type JournalRecord struct {
	LSN     uint64
	Payload []byte
}

// JournalCursor is a follower's position in a specific store's journal. The
// zero StoreID means "whatever this store is"; a non-zero one is checked, so a
// follower that has ever synced can never be repointed at a different store
// without noticing.
type JournalCursor struct {
	StoreID [16]byte
	LSN     uint64
}

// JournalPosition reports the leader's identity and how far its journal has
// been durably written.
type JournalPosition struct {
	StoreID   [16]byte
	LatestLSN uint64
}

var (
	// ErrJournalGap means the records the follower still needs are no longer on
	// disk: a checkpoint has already covered and removed them. The follower must
	// take a fresh snapshot and resume from its AppliedLSN.
	ErrJournalGap = errors.New("collection journal no longer retains the requested LSN")
	// ErrJournalStoreMismatch means the cursor was issued by a different store.
	ErrJournalStoreMismatch = errors.New("collection journal belongs to a different store")
)

// journalNotifier wakes followers when a record is appended. It is a broadcast
// rather than a poll so an idle store costs a follower nothing and a busy one
// does not add a polling interval to replication lag.
//
// It deliberately does not use the store mutex: notify() runs on the append
// path, which already holds it, and wait() must not contend with writers.
type journalNotifier struct {
	mu sync.Mutex
	ch chan struct{}
}

// wait returns a channel closed by the next notify. Callers must obtain it
// BEFORE reading the journal, or an append landing between the read and the
// wait is lost and the follower sleeps until the following one.
func (n *journalNotifier) wait() <-chan struct{} {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.ch == nil {
		n.ch = make(chan struct{})
	}
	return n.ch
}

func (n *journalNotifier) notify() {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.ch != nil {
		close(n.ch)
		n.ch = nil
	}
}

// JournalStatus reports the store's identity and its last durable LSN.
func (s *DurableStore) JournalStatus() (JournalPosition, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return JournalPosition{}, err
	}
	return JournalPosition{StoreID: s.metadata.StoreID, LatestLSN: s.journalLastLSN()}, nil
}

// StreamJournal visits every durable record after cursor.LSN, in order, and
// reports the store's position when the read started.
//
// It holds no store lock while reading, so writes proceed during the scan; the
// reader stops at the file length it observed when it opened each artifact and
// therefore always returns a committed prefix. Records already covered by a
// checkpoint are gone from disk, so a cursor that has fallen behind gets
// ErrJournalGap before any record is delivered rather than a silent hole.
func (s *DurableStore) StreamJournal(cursor JournalCursor, visit func(JournalRecord) error) (JournalPosition, error) {
	s.mu.RLock()
	err := s.stateErrorLocked()
	pos := JournalPosition{StoreID: s.metadata.StoreID, LatestLSN: s.journalLastLSN()}
	currentPath, frozenPath := s.journal.currentPath, s.journal.frozenPath
	maxPayload := s.journal.maxPayload
	s.mu.RUnlock()
	if err != nil {
		return JournalPosition{}, err
	}
	if cursor.StoreID != ([16]byte{}) && cursor.StoreID != pos.StoreID {
		return pos, fmt.Errorf("%w: cursor %x, store %x", ErrJournalStoreMismatch, cursor.StoreID, pos.StoreID)
	}
	// A follower strictly ahead of its leader has diverged, and "caught up" is
	// the one answer that is certainly wrong. StoreID survives a cold restore of
	// the leader's data directory from an older backup, so the mismatch check
	// above cannot see it; answering nil parks the follower on a tail that will
	// never reach it while it serves reads nothing will ever correct, and the
	// per-record successor check never runs because no record is delivered.
	// ErrJournalGap is the right signal even though nothing was pruned: the
	// caller's recovery is identical either way -- stop and re-bootstrap.
	if cursor.LSN > pos.LatestLSN {
		return pos, fmt.Errorf("%w: cursor at LSN %d is ahead of the leader at LSN %d", ErrJournalGap, cursor.LSN, pos.LatestLSN)
	}
	if cursor.LSN == pos.LatestLSN {
		return pos, nil
	}

	var (
		buffer   []byte
		lastSeen = cursor.LSN
	)
	deliver := func(record collectionJournalRecord) error {
		if record.LSN <= lastSeen {
			return nil
		}
		// Records are dense and ordered, so the first one that survives the
		// cursor filter proves whether anything was lost: it must be exactly the
		// successor. This is checked before the visitor runs, so a follower
		// never applies a record on the far side of a hole.
		if record.LSN != lastSeen+1 {
			return fmt.Errorf("%w: cursor at LSN %d, oldest retained is LSN %d", ErrJournalGap, lastSeen, record.LSN)
		}
		if visit != nil {
			if err := visit(JournalRecord{LSN: record.LSN, Payload: record.Payload}); err != nil {
				return err
			}
		}
		lastSeen = record.LSN
		return nil
	}

	// Frozen holds the older half of the range whenever a rotation has happened
	// but its checkpoint has not yet covered it, so it must be read first.
	for _, path := range [2]string{frozenPath, currentPath} {
		if _, err := scanCollectionJournalFile(path, pos.StoreID, maxPayload, nil, &buffer, true, deliver); err != nil {
			var tail *collectionJournalPartialTailError
			if errors.As(err, &tail) {
				// The writer is mid-frame. Everything before it was checksum-
				// verified and has already been delivered; the rest arrives on
				// the next pass. Treating this as corruption would make every
				// concurrent append a replication failure.
				continue
			}
			return pos, err
		}
	}
	// Nothing readable is left, yet the cursor is still behind the position
	// sampled before the scan: a checkpoint removed the records in between.
	// The gap check inside deliver only fires when a record is actually read,
	// so without this the caller is told everything is fine and FollowJournal
	// parks on an idle leader forever -- the silent stall the legacy cluster
	// shipped, where replication stopped permanently and nothing was logged.
	// It cannot false-positive: LatestLSN is only advanced after a frame is
	// written and fsynced, so every LSN up to it is a complete frame on disk,
	// and records appended during the scan only push lastSeen above it.
	if lastSeen < pos.LatestLSN {
		return pos, fmt.Errorf("%w: cursor at LSN %d, leader at LSN %d, nothing retained in between", ErrJournalGap, lastSeen, pos.LatestLSN)
	}
	return pos, nil
}

// FollowJournal streams records after cursor.LSN and then keeps streaming as
// new ones are appended, until ctx is canceled, visit fails, or the stream
// breaks (ErrJournalGap, ErrJournalStoreMismatch, a store fault).
func (s *DurableStore) FollowJournal(ctx context.Context, cursor JournalCursor, visit func(JournalRecord) error) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		// Registered before the read so an append that lands mid-read still
		// wakes this follower.
		wake := s.appended.wait()
		pos, err := s.StreamJournal(cursor, func(record JournalRecord) error {
			if visit != nil {
				if err := visit(record); err != nil {
					return err
				}
			}
			cursor.LSN = record.LSN
			return nil
		})
		if err != nil {
			return err
		}
		cursor.StoreID = pos.StoreID
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-wake:
		}
	}
}

// journalLastLSN reads the writer's durable counter. Callers hold s.mu, which
// excludes appends, but the journal has its own mutex for rotation and cleanup.
func (s *DurableStore) journalLastLSN() uint64 {
	s.journal.mu.Lock()
	defer s.journal.mu.Unlock()
	return s.journal.lastLSN
}

// WriteSnapshot streams one complete snapshot generation of this store to w and
// reports the position those bytes capture.
//
// It is the other half of replica bootstrap: a leader that has checkpointed no
// longer retains the journal records a new follower would need, so the follower
// has to be handed state instead of history. The bytes are exactly what
// Checkpoint commits to disk -- same encoder, same header, same trailing
// checksum -- so BootstrapReplica can hand them straight to the normal open
// path and inherit every validation it does.
//
// LatestLSN is the AppliedLSN encoded in the header, which is the counter the
// receiving store's journal is seeded with, so the first record the follower
// can accept is LatestLSN+1.
//
// The store's write barrier is held for the whole write: w must be a local sink
// (an *os.File, a *bytes.Buffer), never a socket.
//
// ponytail: a remote-paced writer would block every local write for the length
// of the transfer. Upgrade trigger is the first bootstrap that must stream
// straight to a peer -- then snapshot to a temp file under the lock and ship
// the file unlocked.
func (s *DurableStore) WriteSnapshot(w io.Writer) (JournalPosition, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return JournalPosition{}, err
	}
	// Unwritten precondition of the encoder: it ranges both maps and bakes
	// their sizes into the header before writing any frame, so the counts and
	// the frames disagree if either map moves. Same acquisition order as
	// saveUnifiedCollectionSnapshot, under the same s.mu, so no inversion.
	s.manager.mu.RLock()
	defer s.manager.mu.RUnlock()
	s.tenants.mu.RLock()
	defer s.tenants.mu.RUnlock()
	if err := writeUnifiedCollectionSnapshotV2(w, s.manager, s.tenants, s.metadata); err != nil {
		return JournalPosition{}, fmt.Errorf("write collection store snapshot: %w", err)
	}
	// AppliedLSN, not journalLastLSN: it is the value just written into the
	// header. Reporting the journal counter would promise a record the snapshot
	// does not contain and open the follower one record short.
	return JournalPosition{StoreID: s.metadata.StoreID, LatestLSN: s.metadata.AppliedLSN}, nil
}
