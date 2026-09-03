package collection

import (
	"context"
	"errors"
	"fmt"
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
	if cursor.LSN >= pos.LatestLSN {
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
