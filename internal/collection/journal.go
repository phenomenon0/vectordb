package collection

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

const (
	collectionJournalVersion    = uint16(1)
	collectionJournalHeaderSize = uint16(72)
	collectionJournalMaxPayload = uint32(16 << 20)

	collectionJournalVersionOffset    = 8
	collectionJournalHeaderSizeOffset = 10
	collectionJournalStoreIDOffset    = 12
	collectionJournalLSNOffset        = 28
	collectionJournalPayloadLenOffset = 36
	collectionJournalChecksumOffset   = 40
)

var (
	collectionJournalMagic = [8]byte{'D', 'D', 'C', 'O', 'L', 'J', 'N', 'L'}

	errCollectionJournalFaulted      = errors.New("collection journal writes are disabled")
	errCollectionJournalFrozenExists = errors.New("collection journal frozen artifact already exists")
)

// collectionJournalRecord is a validated, durable journal frame. Payload is
// intentionally opaque: the durable collection layer owns mutation semantics.
type collectionJournalRecord struct {
	LSN     uint64
	Payload []byte
}

// collectionJournalRecordVisitor consumes one validated frame synchronously.
// Payload aliases a bounded scanner buffer and is valid only until the visitor
// returns. Callers that need to retain it must make an explicit copy.
type collectionJournalRecordVisitor func(collectionJournalRecord) error

// collectionJournalFileSummary is bounded validation evidence for one
// artifact. It deliberately contains no record payloads.
type collectionJournalFileSummary struct {
	exists      bool
	info        os.FileInfo
	recordCount uint64
	firstLSN    uint64
	lastLSN     uint64
	digest      [sha256.Size]byte
}

// collectionJournalReplayPlan binds a later streaming replay pass to the
// exact artifacts that passed validation. The plan is constant-size no matter
// how many frames or payload bytes the journal contains.
type collectionJournalReplayPlan struct {
	appliedLSN      uint64
	replayRecords   uint64
	firstReplayLSN  uint64
	lastReplayLSN   uint64
	lastArtifactLSN uint64
	frozen          collectionJournalFileSummary
	current         collectionJournalFileSummary
}

// collectionJournalPartialTailError describes an EOF-short final frame. The
// validated prefix is represented only by aggregate evidence so the open path
// can decide whether the current artifact is safe to truncate. Callers must not
// replay it until repair has completed and the artifact has been reparsed.
type collectionJournalPartialTailError struct {
	path        string
	offset      int64
	fileSize    int64
	header      []byte
	summary     collectionJournalFileSummary
	partialBody bool
}

func (e *collectionJournalPartialTailError) Error() string {
	if e.partialBody {
		return fmt.Sprintf("collection journal %q has a partial frame body at offset %d", e.path, e.offset)
	}
	return fmt.Sprintf("collection journal %q has a partial frame header at offset %d", e.path, e.offset)
}

// collectionJournal serializes append, rotation, and cleanup for one store.
// Callers must use openCollectionJournal so existing artifacts are validated
// before the writer can be used.
type collectionJournal struct {
	mu sync.Mutex

	currentPath string
	frozenPath  string
	storeID     [16]byte
	lastLSN     uint64
	maxPayload  uint32
	fault       error
	ops         collectionJournalFileOps

	// file caches an open descriptor for the current artifact so appends do
	// not pay open/chmod/stat/close per record. It is nil until the first
	// append and is invalidated (closed) whenever the artifact identity can
	// change: rotation, covered cleanup, and fault latching.
	file *os.File
	// fileNeedsNameSync records that the directory entry for the current
	// artifact has not yet been made durable by THIS writer. A prior process
	// may have crashed after syncing a new inode but before its directory
	// entry survived, so every freshly opened descriptor syncs the parent
	// directory exactly once before further name syncs are skipped.
	fileNeedsNameSync bool
}

// collectionJournalFileOps provides per-instance fault-injection seams. Keeping
// these hooks off package globals makes race tests and concurrent stores safe.
type collectionJournalFileOps struct {
	openFile      func(string, int, os.FileMode) (*os.File, error)
	writeFile     func(*os.File, []byte) (int, error)
	truncateFile  func(*os.File, int64) error
	chmodFile     func(*os.File, os.FileMode) error
	syncFile      func(*os.File) error
	closeFile     func(*os.File) error
	statPath      func(string) (os.FileInfo, error)
	linkNoReplace func(string, string) error
	removePath    func(string) error
	syncDir       func(string) error
}

func defaultCollectionJournalFileOps() collectionJournalFileOps {
	return collectionJournalFileOps{
		openFile: os.OpenFile,
		writeFile: func(f *os.File, p []byte) (int, error) {
			return f.Write(p)
		},
		truncateFile: func(f *os.File, size int64) error {
			return f.Truncate(size)
		},
		chmodFile: func(f *os.File, mode os.FileMode) error {
			return f.Chmod(mode)
		},
		syncFile: func(f *os.File) error {
			return f.Sync()
		},
		closeFile: func(f *os.File) error {
			return f.Close()
		},
		statPath:      os.Stat,
		linkNoReplace: os.Link,
		removePath:    os.Remove,
		syncDir:       syncCollectionJournalDir,
	}
}

// openCollectionJournal validates frozen then current artifacts in full before
// returning a usable writer. appliedLSN is the high-water mark already present
// in the caller's snapshot. The returned replay plan contains only aggregate
// validation evidence; payloads are never retained by the open path.
func openCollectionJournal(currentPath, frozenPath string, storeID [16]byte, appliedLSN uint64) (*collectionJournal, collectionJournalReplayPlan, error) {
	return openCollectionJournalValidated(currentPath, frozenPath, storeID, appliedLSN, nil)
}

// openCollectionJournalValidated additionally invokes validate for every
// replayable record during the full validation pass. This is the startup seam
// used by DurableStore to decode every mutation before any loaded state is
// changed. Validation remains bounded to one maximum-sized record.
func openCollectionJournalValidated(
	currentPath, frozenPath string,
	storeID [16]byte,
	appliedLSN uint64,
	validate collectionJournalRecordVisitor,
) (*collectionJournal, collectionJournalReplayPlan, error) {
	j, err := newCollectionJournal(currentPath, frozenPath, storeID, appliedLSN)
	if err != nil {
		return nil, collectionJournalReplayPlan{}, err
	}

	plan, err := j.validateAndRepair(appliedLSN, validate)
	if err != nil {
		return nil, collectionJournalReplayPlan{}, err
	}
	return j, plan, nil
}

func newCollectionJournal(currentPath, frozenPath string, storeID [16]byte, lastLSN uint64) (*collectionJournal, error) {
	currentPath = filepath.Clean(currentPath)
	frozenPath = filepath.Clean(frozenPath)
	if currentPath == "." || frozenPath == "." {
		return nil, errors.New("collection journal paths cannot be empty")
	}
	if currentPath == frozenPath {
		return nil, errors.New("collection journal current and frozen paths must differ")
	}
	if filepath.Dir(currentPath) != filepath.Dir(frozenPath) {
		return nil, errors.New("collection journal current and frozen paths must share a directory")
	}
	if storeID == ([16]byte{}) {
		return nil, errors.New("collection journal store UUID cannot be zero")
	}

	return &collectionJournal{
		currentPath: currentPath,
		frozenPath:  frozenPath,
		storeID:     storeID,
		lastLSN:     lastLSN,
		maxPayload:  collectionJournalMaxPayload,
		ops:         defaultCollectionJournalFileOps(),
	}, nil
}

// append writes one complete frame and fsyncs it before reporting success. Any
// error after frame I/O begins is treated as indeterminate and permanently
// faults this writer; recovery must reopen and validate the journal.
//
// On success the returned record carries only the assigned LSN; payloads are
// intentionally not returned. Replay via streamReplay is the sole source for
// previously appended payload bytes.
//
// Durability semantics per record: the frame bytes and the file metadata are
// synced on every append (fsync), while the parent directory is synced once
// per open descriptor. Directory entries change only when an artifact name is
// created or replaced, so after this writer has itself persisted the current
// name, later appends to the same inode need no further namespace barrier.
func (j *collectionJournal) append(payload []byte) (collectionJournalRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()

	if j.fault != nil {
		return collectionJournalRecord{}, fmt.Errorf("%w: %v", errCollectionJournalFaulted, j.fault)
	}
	if uint64(len(payload)) > uint64(j.maxPayload) {
		return collectionJournalRecord{}, fmt.Errorf("collection journal payload is %d bytes; maximum is %d", len(payload), j.maxPayload)
	}
	if j.lastLSN == math.MaxUint64 {
		return collectionJournalRecord{}, errors.New("collection journal LSN exhausted")
	}

	lsn := j.lastLSN + 1
	frame, err := encodeCollectionJournalFrame(j.storeID, lsn, payload, j.maxPayload)
	if err != nil {
		return collectionJournalRecord{}, err
	}

	if j.file == nil {
		f, _, openErr := j.openAppendFile()
		if openErr != nil {
			return collectionJournalRecord{}, fmt.Errorf("open collection journal: %w", openErr)
		}
		j.file = f
		// The durability of the artifact's directory entry is unknown for a
		// descriptor this process did not sync yet; establish it once.
		j.fileNeedsNameSync = true
	}

	writeErr := writeCollectionJournalFrame(j.ops.writeFile, j.file, frame)
	var syncErr error
	if writeErr == nil {
		syncErr = j.ops.syncFile(j.file)
	}
	var dirErr error
	if writeErr == nil && syncErr == nil && j.fileNeedsNameSync {
		// Sync the namespace when the artifact name may not be durable. See
		// the field comment: a surviving-but-unnamed inode would make any
		// acknowledged frame unrecoverable after a crash.
		dirErr = j.ops.syncDir(filepath.Dir(j.currentPath))
		if dirErr == nil {
			j.fileNeedsNameSync = false
		}
	}

	if operationErr := errors.Join(writeErr, syncErr, dirErr); operationErr != nil {
		j.fault = fmt.Errorf("collection journal append at LSN %d is indeterminate: %w", lsn, operationErr)
		closeErr := j.closeWriterLocked()
		if closeErr != nil {
			j.fault = errors.Join(j.fault, closeErr)
		}
		return collectionJournalRecord{}, j.fault
	}

	j.lastLSN = lsn
	return collectionJournalRecord{LSN: lsn}, nil
}

// closeWriterLocked closes and forgets any cached descriptor. Callers hold
// j.mu. It is invoked whenever the current artifact's identity changes or the
// writer shuts down; the next append reopens and re-establishes namespace
// durability from scratch.
func (j *collectionJournal) closeWriterLocked() error {
	if j.file == nil {
		return nil
	}
	f := j.file
	j.file = nil
	j.fileNeedsNameSync = false
	return j.ops.closeFile(f)
}

func (j *collectionJournal) openAppendFile() (*os.File, bool, error) {
	const flags = os.O_WRONLY | os.O_APPEND
	f, err := j.ops.openFile(j.currentPath, flags|os.O_CREATE|os.O_EXCL, 0o600)
	created := err == nil
	if err != nil {
		if !errors.Is(err, os.ErrExist) {
			return nil, false, err
		}
		f, err = j.ops.openFile(j.currentPath, flags, 0)
		if err != nil {
			return nil, false, err
		}
	}

	// O_CREATE respects umask but does not repair an existing file. Force the
	// invariant on every open so journal payloads are never group/world-readable.
	if err := j.ops.chmodFile(f, 0o600); err != nil {
		closeErr := j.ops.closeFile(f)
		return nil, created, errors.Join(err, closeErr)
	}
	info, err := f.Stat()
	if err != nil {
		closeErr := j.ops.closeFile(f)
		return nil, created, errors.Join(err, closeErr)
	}
	if !info.Mode().IsRegular() {
		closeErr := j.ops.closeFile(f)
		return nil, created, errors.Join(errors.New("collection journal is not a regular file"), closeErr)
	}
	return f, created, nil
}

// collectionJournalSequenceState validates the ordering contract across a
// frozen/current scan without retaining any frame. Covered records may overlap
// between artifacts for compatibility with older checkpoints, but replayable
// records must form one exact sequence beginning at appliedLSN+1.
type collectionJournalSequenceState struct {
	appliedLSN      uint64
	replayRecords   uint64
	firstReplayLSN  uint64
	lastReplayLSN   uint64
	lastArtifactLSN uint64
}

func (s *collectionJournalSequenceState) accept(
	path string,
	record collectionJournalRecord,
	previousInArtifact uint64,
	hasPreviousInArtifact bool,
) error {
	if hasPreviousInArtifact {
		switch {
		case record.LSN <= previousInArtifact:
			return fmt.Errorf("collection journal LSN %d in %q duplicates or regresses from %d", record.LSN, path, previousInArtifact)
		case previousInArtifact == math.MaxUint64 || record.LSN != previousInArtifact+1:
			return fmt.Errorf("collection journal LSN gap in %q: got %d after %d", path, record.LSN, previousInArtifact)
		}
	}
	if record.LSN > s.lastArtifactLSN {
		s.lastArtifactLSN = record.LSN
	}
	if record.LSN <= s.appliedLSN {
		if s.replayRecords > 0 {
			return fmt.Errorf("collection journal LSN %d in %q duplicates or regresses behind replay after checkpoint %d", record.LSN, path, s.appliedLSN)
		}
		return nil
	}

	if s.replayRecords == 0 {
		if s.appliedLSN == math.MaxUint64 || record.LSN != s.appliedLSN+1 {
			return fmt.Errorf("collection journal LSN gap after checkpoint %d: first replay record is %d", s.appliedLSN, record.LSN)
		}
		s.firstReplayLSN = record.LSN
	} else {
		previous := s.lastReplayLSN
		if record.LSN <= previous {
			return fmt.Errorf("collection journal LSN %d in %q duplicates or regresses from %d", record.LSN, path, previous)
		}
		if previous == math.MaxUint64 || record.LSN != previous+1 {
			return fmt.Errorf("collection journal LSN gap during replay: got %d after %d", record.LSN, previous)
		}
	}
	s.replayRecords++
	s.lastReplayLSN = record.LSN
	return nil
}

// validateAndRepair performs the fail-closed startup pass. It validates frame
// structure, checksums, store identity, LSN order, and (through validate) the
// complete mutation encoding before a caller is allowed to mutate loaded
// collection state. Only one payload-sized buffer is live at any time.
func (j *collectionJournal) validateAndRepair(appliedLSN uint64, validate collectionJournalRecordVisitor) (collectionJournalReplayPlan, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if err := j.repairInterruptedRotation(); err != nil {
		return collectionJournalReplayPlan{}, err
	}

	var payloadBuffer []byte
	state := collectionJournalSequenceState{appliedLSN: appliedLSN}
	frozen, err := scanCollectionJournalArtifact(
		j.frozenPath,
		j.storeID,
		j.maxPayload,
		nil,
		&payloadBuffer,
		&state,
		validate,
	)
	if err != nil {
		return collectionJournalReplayPlan{}, err
	}
	stateAfterFrozen := state

	current, err := scanCollectionJournalArtifact(
		j.currentPath,
		j.storeID,
		j.maxPayload,
		nil,
		&payloadBuffer,
		&state,
		validate,
	)
	if err != nil {
		var partial *collectionJournalPartialTailError
		if !errors.As(err, &partial) {
			return collectionJournalReplayPlan{}, err
		}

		expectedLSN, tailErr := expectedCollectionJournalTailLSN(
			appliedLSN,
			stateAfterFrozen.lastArtifactLSN,
			partial.summary,
		)
		if tailErr != nil {
			return collectionJournalReplayPlan{}, tailErr
		}
		if tailErr := validateCollectionJournalPartialTail(partial, j.storeID, expectedLSN, j.maxPayload); tailErr != nil {
			return collectionJournalReplayPlan{}, fmt.Errorf("refuse collection journal partial-tail repair: %w", tailErr)
		}
		if tailErr := rejectCompleteFrameAsPartialTail(partial); tailErr != nil {
			return collectionJournalReplayPlan{}, fmt.Errorf("refuse collection journal partial-tail repair: %w", tailErr)
		}
		if tailErr := j.repairCurrentPartialTail(partial); tailErr != nil {
			return collectionJournalReplayPlan{}, tailErr
		}

		// Reparse the repaired artifact from byte zero. The validation callback
		// intentionally runs again so no bytes trusted before the durable change
		// can authorize replay afterward.
		state = stateAfterFrozen
		current, err = scanCollectionJournalArtifact(
			j.currentPath,
			j.storeID,
			j.maxPayload,
			nil,
			&payloadBuffer,
			&state,
			validate,
		)
		if err != nil {
			return collectionJournalReplayPlan{}, fmt.Errorf("reparse repaired current collection journal: %w", err)
		}
	}

	if state.lastArtifactLSN > j.lastLSN {
		j.lastLSN = state.lastArtifactLSN
	}
	// A stale current prefix ending before the checkpoint cannot safely accept
	// appliedLSN+1 because doing so would create an intra-file gap. Its contents
	// are already covered, so durably remove it and bind the plan to its absence.
	if current.exists && current.recordCount > 0 && current.lastLSN < appliedLSN {
		if err := j.ops.removePath(j.currentPath); err != nil {
			return collectionJournalReplayPlan{}, fmt.Errorf("remove checkpoint-covered current collection journal: %w", err)
		}
		if err := j.ops.syncDir(filepath.Dir(j.currentPath)); err != nil {
			return collectionJournalReplayPlan{}, fmt.Errorf("sync checkpoint-covered current journal cleanup: %w", err)
		}
		current = collectionJournalFileSummary{}
		state = stateAfterFrozen
	}

	return replayPlanFromState(appliedLSN, state, frozen, current), nil
}

// streamReplay reopens the exact artifacts captured by plan and passes one
// validated replay frame at a time to visit. Payload is never retained here;
// it aliases a single reusable buffer. Artifact identity and metadata are
// checked before visiting that artifact, and its content digest is rechecked
// after the scan as a final integrity guard.
func (j *collectionJournal) streamReplay(plan collectionJournalReplayPlan, visit collectionJournalRecordVisitor) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	var payloadBuffer []byte
	state := collectionJournalSequenceState{appliedLSN: plan.appliedLSN}
	frozen, err := scanCollectionJournalArtifact(
		j.frozenPath,
		j.storeID,
		j.maxPayload,
		&plan.frozen,
		&payloadBuffer,
		&state,
		visit,
	)
	if err != nil {
		return fmt.Errorf("reopen validated frozen collection journal: %w", err)
	}
	current, err := scanCollectionJournalArtifact(
		j.currentPath,
		j.storeID,
		j.maxPayload,
		&plan.current,
		&payloadBuffer,
		&state,
		visit,
	)
	if err != nil {
		return fmt.Errorf("reopen validated current collection journal: %w", err)
	}
	actual := replayPlanFromState(plan.appliedLSN, state, frozen, current)
	if actual.replayRecords != plan.replayRecords ||
		actual.firstReplayLSN != plan.firstReplayLSN ||
		actual.lastReplayLSN != plan.lastReplayLSN ||
		actual.lastArtifactLSN != plan.lastArtifactLSN {
		return errors.New("collection journal replay sequence changed after validation")
	}
	return nil
}

// verifyCovered streams and validates both artifacts while retaining only
// aggregate coverage evidence. It is used after a snapshot commit and before
// journal cleanup.
func (j *collectionJournal) verifyCovered(appliedLSN uint64) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	var (
		payloadBuffer []byte
		state         = collectionJournalSequenceState{appliedLSN: appliedLSN}
		uncoveredPath string
		uncoveredLSN  uint64
	)
	for _, path := range []string{j.frozenPath, j.currentPath} {
		_, err := scanCollectionJournalArtifact(
			path,
			j.storeID,
			j.maxPayload,
			nil,
			&payloadBuffer,
			&state,
			func(record collectionJournalRecord) error {
				if uncoveredPath == "" {
					uncoveredPath = path
					uncoveredLSN = record.LSN
				}
				return nil
			},
		)
		if err != nil {
			return fmt.Errorf("verify checkpoint coverage for %q: %w", path, err)
		}
	}
	if uncoveredPath != "" {
		return fmt.Errorf("refusing journal cleanup: %q LSN %d exceeds snapshot LSN %d", uncoveredPath, uncoveredLSN, appliedLSN)
	}
	return nil
}

func replayPlanFromState(
	appliedLSN uint64,
	state collectionJournalSequenceState,
	frozen collectionJournalFileSummary,
	current collectionJournalFileSummary,
) collectionJournalReplayPlan {
	return collectionJournalReplayPlan{
		appliedLSN:      appliedLSN,
		replayRecords:   state.replayRecords,
		firstReplayLSN:  state.firstReplayLSN,
		lastReplayLSN:   state.lastReplayLSN,
		lastArtifactLSN: state.lastArtifactLSN,
		frozen:          frozen,
		current:         current,
	}
}

func expectedCollectionJournalTailLSN(
	appliedLSN uint64,
	previousArtifactLSN uint64,
	current collectionJournalFileSummary,
) (uint64, error) {
	lastLSN := appliedLSN
	if previousArtifactLSN > lastLSN {
		lastLSN = previousArtifactLSN
	}
	if current.recordCount > 0 {
		lastLSN = current.lastLSN
	}
	if lastLSN == math.MaxUint64 {
		return 0, errors.New("collection journal LSN exhausted before partial tail")
	}
	return lastLSN + 1, nil
}

func validateCollectionJournalPartialTail(
	tail *collectionJournalPartialTailError,
	storeID [16]byte,
	expectedLSN uint64,
	maxPayload uint32,
) error {
	if tail == nil || len(tail.header) == 0 {
		return errors.New("partial journal tail has no header prefix")
	}
	if tail.offset < 0 || tail.fileSize <= tail.offset {
		return errors.New("partial journal tail has invalid file bounds")
	}
	if tail.partialBody && len(tail.header) != int(collectionJournalHeaderSize) {
		return errors.New("partial journal body is missing its complete header")
	}
	if !tail.partialBody && len(tail.header) >= int(collectionJournalHeaderSize) {
		return errors.New("partial journal header is not EOF-short")
	}

	fixed := make([]byte, collectionJournalPayloadLenOffset)
	copy(fixed[:len(collectionJournalMagic)], collectionJournalMagic[:])
	binary.BigEndian.PutUint16(fixed[collectionJournalVersionOffset:], collectionJournalVersion)
	binary.BigEndian.PutUint16(fixed[collectionJournalHeaderSizeOffset:], collectionJournalHeaderSize)
	copy(fixed[collectionJournalStoreIDOffset:collectionJournalLSNOffset], storeID[:])
	binary.BigEndian.PutUint64(fixed[collectionJournalLSNOffset:], expectedLSN)
	fixedPrefixLen := len(tail.header)
	if fixedPrefixLen > len(fixed) {
		fixedPrefixLen = len(fixed)
	}
	if !bytes.Equal(tail.header[:fixedPrefixLen], fixed[:fixedPrefixLen]) {
		return fmt.Errorf("partial journal header does not match the expected frame prefix for LSN %d", expectedLSN)
	}

	if len(tail.header) > collectionJournalPayloadLenOffset {
		available := len(tail.header) - collectionJournalPayloadLenOffset
		if available > 4 {
			available = 4
		}
		var payloadLengthPrefix [4]byte
		copy(payloadLengthPrefix[:available], tail.header[collectionJournalPayloadLenOffset:collectionJournalPayloadLenOffset+available])
		if minimumPayloadLength := binary.BigEndian.Uint32(payloadLengthPrefix[:]); minimumPayloadLength > maxPayload {
			return fmt.Errorf("partial journal header cannot encode a payload within maximum %d", maxPayload)
		}
	}
	return nil
}

// rejectCompleteFrameAsPartialTail guards truncation against an acknowledged
// frame whose payload-length field was corrupted upward past EOF. By framing
// alone that is indistinguishable from a torn write, so the body is checked:
// if the stored checksum verifies for a candidate length ending at EOF or at a
// successor frame's magic, the frame was complete and truncating it would
// discard acknowledged records. The body is bounded by maxPayload because the
// scanner already rejected larger claimed lengths.
func rejectCompleteFrameAsPartialTail(tail *collectionJournalPartialTailError) error {
	if !tail.partialBody {
		return nil
	}
	f, err := os.Open(tail.path)
	if err != nil {
		return fmt.Errorf("open partial journal tail body: %w", err)
	}
	defer f.Close()
	bodyOffset := tail.offset + int64(collectionJournalHeaderSize)
	body := make([]byte, tail.fileSize-bodyOffset)
	if _, err := f.ReadAt(body, bodyOffset); err != nil {
		return fmt.Errorf("read partial journal tail body: %w", err)
	}

	header := append([]byte(nil), tail.header[:collectionJournalChecksumOffset]...)
	stored := tail.header[collectionJournalChecksumOffset:int(collectionJournalHeaderSize)]
	h := sha256.New()
	completeWithLength := func(payloadLen int) bool {
		binary.BigEndian.PutUint32(header[collectionJournalPayloadLenOffset:], uint32(payloadLen))
		h.Reset()
		_, _ = h.Write(header)
		_, _ = h.Write(body[:payloadLen])
		return equalCollectionJournalChecksum(stored, h.Sum(nil))
	}
	candidate := len(body)
	for candidate >= 0 {
		if completeWithLength(candidate) {
			return fmt.Errorf("partial journal tail at offset %d is a complete frame with a corrupt payload length", tail.offset)
		}
		next := bytes.LastIndex(body[:candidate], collectionJournalMagic[:])
		if next < 0 {
			break
		}
		candidate = next
	}
	return nil
}

func (j *collectionJournal) repairCurrentPartialTail(tail *collectionJournalPartialTailError) error {
	if tail == nil || tail.path != j.currentPath {
		return errors.New("refusing to repair a non-current collection journal tail")
	}
	f, err := j.ops.openFile(j.currentPath, os.O_RDWR, 0)
	if err != nil {
		return fmt.Errorf("open current collection journal for partial-tail repair: %w", err)
	}
	info, statErr := f.Stat()
	if statErr == nil {
		switch {
		case !info.Mode().IsRegular():
			statErr = errors.New("current collection journal is not a regular file")
		case info.Mode().Perm() != 0o600:
			statErr = fmt.Errorf("current collection journal has permissions %04o; expected 0600", info.Mode().Perm())
		case tail.summary.info != nil && !os.SameFile(tail.summary.info, info):
			statErr = errors.New("current collection journal was replaced before partial-tail repair")
		case info.Size() != tail.fileSize:
			statErr = fmt.Errorf("current collection journal changed size from %d to %d before partial-tail repair", tail.fileSize, info.Size())
		}
	}
	if statErr != nil {
		return errors.Join(fmt.Errorf("validate current collection journal before partial-tail repair: %w", statErr), j.ops.closeFile(f))
	}

	truncateErr := j.ops.truncateFile(f, tail.offset)
	var syncErr error
	if truncateErr == nil {
		syncErr = j.ops.syncFile(f)
	}
	closeErr := j.ops.closeFile(f)
	var dirErr error
	if truncateErr == nil {
		dirErr = j.ops.syncDir(filepath.Dir(j.currentPath))
	}
	if repairErr := errors.Join(truncateErr, syncErr, closeErr, dirErr); repairErr != nil {
		return fmt.Errorf("repair current collection journal partial tail: %w", repairErr)
	}
	return nil
}

func (j *collectionJournal) repairInterruptedRotation() error {
	frozenInfo, frozenErr := j.ops.statPath(j.frozenPath)
	if frozenErr != nil {
		if errors.Is(frozenErr, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("stat frozen collection journal during rotation recovery: %w", frozenErr)
	}
	currentInfo, currentErr := j.ops.statPath(j.currentPath)
	if currentErr != nil {
		if errors.Is(currentErr, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("stat current collection journal during rotation recovery: %w", currentErr)
	}
	if !os.SameFile(frozenInfo, currentInfo) {
		return nil
	}
	// The link may have been created immediately before a crash and therefore
	// may not be durable yet. Persist it before unlinking the old name.
	if err := j.ops.syncDir(filepath.Dir(j.currentPath)); err != nil {
		return fmt.Errorf("sync frozen link during interrupted collection journal rotation recovery: %w", err)
	}
	if err := j.ops.removePath(j.currentPath); err != nil {
		return fmt.Errorf("complete interrupted collection journal rotation: %w", err)
	}
	if err := j.ops.syncDir(filepath.Dir(j.currentPath)); err != nil {
		return fmt.Errorf("sync interrupted collection journal rotation recovery: %w", err)
	}
	return nil
}

// rotate moves current to frozen without replacing an existing artifact. A
// portable hard-link protocol uses a directory sync before and after unlinking
// the old name so a crash can never lose both names.
func (j *collectionJournal) rotate() error {
	j.mu.Lock()
	defer j.mu.Unlock()

	if j.fault != nil {
		return fmt.Errorf("%w: %v", errCollectionJournalFaulted, j.fault)
	}
	// The cached descriptor belongs to the artifact whose name is about to be
	// replaced. Writing through it after the link/unlink would extend the
	// frozen-linked inode, so it must be dropped before any name change. An
	// uncertain close is treated exactly like other indeterminate I/O.
	if err := j.closeWriterLocked(); err != nil {
		j.fault = fmt.Errorf("collection journal descriptor close before rotation is indeterminate: %w", err)
		return j.fault
	}
	if _, err := j.ops.statPath(j.frozenPath); err == nil {
		return fmt.Errorf("%w: %s", errCollectionJournalFrozenExists, j.frozenPath)
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stat frozen collection journal: %w", err)
	}
	if _, err := j.ops.statPath(j.currentPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("stat current collection journal: %w", err)
	}

	if err := j.ops.linkNoReplace(j.currentPath, j.frozenPath); err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("%w: %s", errCollectionJournalFrozenExists, j.frozenPath)
		}
		j.fault = fmt.Errorf("collection journal rotation link is indeterminate: %w", err)
		return j.fault
	}
	if err := j.ops.syncDir(filepath.Dir(j.currentPath)); err != nil {
		j.fault = fmt.Errorf("collection journal frozen-link directory sync is indeterminate: %w", err)
		return j.fault
	}
	if err := j.ops.removePath(j.currentPath); err != nil {
		j.fault = fmt.Errorf("collection journal current-name removal is indeterminate: %w", err)
		return j.fault
	}
	if err := j.ops.syncDir(filepath.Dir(j.currentPath)); err != nil {
		j.fault = fmt.Errorf("collection journal current-name removal sync is indeterminate: %w", err)
		return j.fault
	}
	return nil
}

// cleanupFrozen durably removes the frozen artifact after its high-water mark
// has been committed to a snapshot. If unlink succeeds but directory sync fails,
// a retry still syncs the directory even though the name is already absent.
func (j *collectionJournal) cleanupFrozen() error {
	j.mu.Lock()
	defer j.mu.Unlock()

	removed := true
	if err := j.ops.removePath(j.frozenPath); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove frozen collection journal: %w", err)
		}
		removed = false
	}

	dir := filepath.Dir(j.frozenPath)
	if !removed {
		if _, err := j.ops.statPath(dir); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return fmt.Errorf("stat collection journal directory: %w", err)
		}
	}
	if err := j.ops.syncDir(dir); err != nil {
		return fmt.Errorf("sync collection journal cleanup: %w", err)
	}
	return nil
}

// cleanupAll durably removes both artifacts after a checkpoint is known to
// cover their complete LSN range. It always syncs an existing parent directory,
// including on retries where either or both names have already disappeared.
// Removal errors are collected so one stuck artifact does not prevent the other
// covered artifact from being unlinked and made durable.
func (j *collectionJournal) cleanupAll() error {
	j.mu.Lock()
	defer j.mu.Unlock()

	// Both artifact names are about to disappear; the cached descriptor would
	// keep writing into an unlinked inode if reused. An uncertain close is
	// treated exactly like other indeterminate I/O.
	if err := j.closeWriterLocked(); err != nil {
		j.fault = fmt.Errorf("collection journal descriptor close before cleanup is indeterminate: %w", err)
		return j.fault
	}

	var removeErrs []error
	for _, path := range []string{j.frozenPath, j.currentPath} {
		if err := j.ops.removePath(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			removeErrs = append(removeErrs, fmt.Errorf("remove collection journal %q: %w", path, err))
		}
	}

	dir := filepath.Dir(j.currentPath)
	if _, err := j.ops.statPath(dir); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return errors.Join(removeErrs...)
		}
		removeErrs = append(removeErrs, fmt.Errorf("stat collection journal directory: %w", err))
		return errors.Join(removeErrs...)
	}
	if err := j.ops.syncDir(dir); err != nil {
		removeErrs = append(removeErrs, fmt.Errorf("sync collection journal cleanup: %w", err))
	}
	return errors.Join(removeErrs...)
}

func (j *collectionJournal) writeFault() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.fault
}

// closeWriter releases the cached descriptor. It is called during graceful
// shutdown and abort, after all appends have completed.
func (j *collectionJournal) closeWriter() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.closeWriterLocked()
}

func encodeCollectionJournalFrame(storeID [16]byte, lsn uint64, payload []byte, maxPayload uint32) ([]byte, error) {
	if storeID == ([16]byte{}) {
		return nil, errors.New("collection journal store UUID cannot be zero")
	}
	if lsn == 0 {
		return nil, errors.New("collection journal LSN cannot be zero")
	}
	if uint64(len(payload)) > uint64(maxPayload) {
		return nil, fmt.Errorf("collection journal payload is %d bytes; maximum is %d", len(payload), maxPayload)
	}

	frame := make([]byte, int(collectionJournalHeaderSize)+len(payload))
	copy(frame[:len(collectionJournalMagic)], collectionJournalMagic[:])
	binary.BigEndian.PutUint16(frame[collectionJournalVersionOffset:], collectionJournalVersion)
	binary.BigEndian.PutUint16(frame[collectionJournalHeaderSizeOffset:], collectionJournalHeaderSize)
	copy(frame[collectionJournalStoreIDOffset:collectionJournalLSNOffset], storeID[:])
	binary.BigEndian.PutUint64(frame[collectionJournalLSNOffset:], lsn)
	binary.BigEndian.PutUint32(frame[collectionJournalPayloadLenOffset:], uint32(len(payload)))
	copy(frame[int(collectionJournalHeaderSize):], payload)

	h := sha256.New()
	_, _ = h.Write(frame[:collectionJournalChecksumOffset])
	_, _ = h.Write(payload)
	copy(frame[collectionJournalChecksumOffset:int(collectionJournalHeaderSize)], h.Sum(nil))
	return frame, nil
}

func scanCollectionJournalArtifact(
	path string,
	storeID [16]byte,
	maxPayload uint32,
	expected *collectionJournalFileSummary,
	payloadBuffer *[]byte,
	state *collectionJournalSequenceState,
	visit collectionJournalRecordVisitor,
) (collectionJournalFileSummary, error) {
	var (
		previousInArtifact    uint64
		hasPreviousInArtifact bool
	)
	return scanCollectionJournalFile(path, storeID, maxPayload, expected, payloadBuffer, func(record collectionJournalRecord) error {
		if err := state.accept(path, record, previousInArtifact, hasPreviousInArtifact); err != nil {
			return err
		}
		previousInArtifact = record.LSN
		hasPreviousInArtifact = true
		if record.LSN > state.appliedLSN && visit != nil {
			return visit(record)
		}
		return nil
	})
}

// scanCollectionJournalFile validates one artifact with constant aggregate
// state and one caller-owned payload buffer. The visitor must consume payload
// synchronously because the next frame overwrites the same backing storage.
func scanCollectionJournalFile(
	path string,
	storeID [16]byte,
	maxPayload uint32,
	expected *collectionJournalFileSummary,
	payloadBuffer *[]byte,
	visit collectionJournalRecordVisitor,
) (collectionJournalFileSummary, error) {
	var summary collectionJournalFileSummary
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			if expected != nil && expected.exists {
				return summary, fmt.Errorf("collection journal %q disappeared after validation", path)
			}
			return summary, nil
		}
		return summary, fmt.Errorf("open collection journal %q: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return summary, fmt.Errorf("stat collection journal %q: %w", path, err)
	}
	summary.exists = true
	summary.info = info
	if !info.Mode().IsRegular() {
		return summary, fmt.Errorf("collection journal %q is not a regular file", path)
	}
	if info.Mode().Perm() != 0o600 {
		return summary, fmt.Errorf("collection journal %q has permissions %04o; expected 0600", path, info.Mode().Perm())
	}
	if err := validateCollectionJournalArtifactIdentity(path, info, expected); err != nil {
		return summary, err
	}
	if payloadBuffer == nil {
		payloadBuffer = new([]byte)
	}

	fileSize := info.Size()
	var (
		offset       int64
		header       [collectionJournalHeaderSize]byte
		artifactHash = sha256.New()
		checksumHash = sha256.New()
	)
	setDigest := func() {
		copy(summary.digest[:], artifactHash.Sum(nil))
	}
	for offset < fileSize {
		remaining := fileSize - offset
		if remaining < int64(collectionJournalHeaderSize) {
			headerPrefix := header[:int(remaining)]
			if _, err := io.ReadFull(f, headerPrefix); err != nil {
				return summary, fmt.Errorf("read collection journal %q partial frame header at offset %d: %w", path, offset, err)
			}
			setDigest()
			return summary, &collectionJournalPartialTailError{
				path:     path,
				offset:   offset,
				fileSize: fileSize,
				header:   append([]byte(nil), headerPrefix...),
				summary:  summary,
			}
		}

		headerBytes := header[:]
		if _, err := io.ReadFull(f, headerBytes); err != nil {
			return summary, fmt.Errorf("read collection journal %q frame header at offset %d: %w", path, offset, err)
		}
		if !bytes.Equal(headerBytes[:len(collectionJournalMagic)], collectionJournalMagic[:]) {
			return summary, fmt.Errorf("collection journal %q has invalid magic at offset %d", path, offset)
		}
		version := binary.BigEndian.Uint16(headerBytes[collectionJournalVersionOffset:])
		if version != collectionJournalVersion {
			return summary, fmt.Errorf("collection journal %q has unknown version %d at offset %d", path, version, offset)
		}
		headerSize := binary.BigEndian.Uint16(headerBytes[collectionJournalHeaderSizeOffset:])
		if headerSize != collectionJournalHeaderSize {
			return summary, fmt.Errorf("collection journal %q has unsupported header size %d at offset %d", path, headerSize, offset)
		}
		var frameStoreID [16]byte
		copy(frameStoreID[:], headerBytes[collectionJournalStoreIDOffset:collectionJournalLSNOffset])
		if frameStoreID != storeID {
			return summary, fmt.Errorf("collection journal %q store UUID mismatch at offset %d", path, offset)
		}
		lsn := binary.BigEndian.Uint64(headerBytes[collectionJournalLSNOffset:])
		if lsn == 0 {
			return summary, fmt.Errorf("collection journal %q has zero LSN at offset %d", path, offset)
		}
		payloadLen := binary.BigEndian.Uint32(headerBytes[collectionJournalPayloadLenOffset:])
		if payloadLen > maxPayload {
			return summary, fmt.Errorf("collection journal %q payload at offset %d is %d bytes; maximum is %d", path, offset, payloadLen, maxPayload)
		}
		if int64(payloadLen) > remaining-int64(collectionJournalHeaderSize) {
			setDigest()
			return summary, &collectionJournalPartialTailError{
				path:        path,
				offset:      offset,
				fileSize:    fileSize,
				header:      append([]byte(nil), headerBytes...),
				summary:     summary,
				partialBody: true,
			}
		}

		if cap(*payloadBuffer) < int(payloadLen) {
			*payloadBuffer = make([]byte, int(payloadLen))
		}
		payload := (*payloadBuffer)[:int(payloadLen)]
		if _, err := io.ReadFull(f, payload); err != nil {
			return summary, fmt.Errorf("read collection journal %q frame body at offset %d: %w", path, offset, err)
		}
		checksumHash.Reset()
		_, _ = checksumHash.Write(headerBytes[:collectionJournalChecksumOffset])
		_, _ = checksumHash.Write(payload)
		if !equalCollectionJournalChecksum(headerBytes[collectionJournalChecksumOffset:int(collectionJournalHeaderSize)], checksumHash.Sum(nil)) {
			return summary, fmt.Errorf("collection journal %q checksum mismatch at offset %d", path, offset)
		}

		_, _ = artifactHash.Write(headerBytes)
		_, _ = artifactHash.Write(payload)
		summary.recordCount++
		if summary.recordCount == 1 {
			summary.firstLSN = lsn
		}
		summary.lastLSN = lsn
		if visit != nil {
			if err := visit(collectionJournalRecord{LSN: lsn, Payload: payload}); err != nil {
				return summary, err
			}
		}
		offset += int64(collectionJournalHeaderSize) + int64(payloadLen)
	}

	// Guard against external growth between Stat and parsing. The journal's own
	// mutex excludes in-process appends, but strict readers must not silently
	// ignore bytes written by another process.
	var extra [1]byte
	if n, err := f.Read(extra[:]); n != 0 || err != io.EOF {
		if err == nil {
			err = errors.New("unexpected trailing byte")
		}
		return summary, fmt.Errorf("collection journal %q changed or has trailing junk after validation: %w", path, err)
	}
	setDigest()

	// Confirm the path still names the inode just scanned and that no metadata
	// relevant to bounded parsing changed while it was open.
	pathInfo, err := os.Stat(path)
	if err != nil {
		return summary, fmt.Errorf("restat collection journal %q after validation: %w", path, err)
	}
	if !os.SameFile(info, pathInfo) || info.Size() != pathInfo.Size() || info.Mode() != pathInfo.Mode() || !info.ModTime().Equal(pathInfo.ModTime()) {
		return summary, fmt.Errorf("collection journal %q changed during validation", path)
	}
	if err := validateCollectionJournalSummary(path, summary, expected); err != nil {
		return summary, err
	}
	return summary, nil
}

func validateCollectionJournalArtifactIdentity(path string, info os.FileInfo, expected *collectionJournalFileSummary) error {
	if expected == nil {
		return nil
	}
	if !expected.exists {
		return fmt.Errorf("collection journal %q appeared after validation", path)
	}
	if expected.info == nil || !os.SameFile(expected.info, info) {
		return fmt.Errorf("collection journal %q was replaced after validation", path)
	}
	if expected.info.Size() != info.Size() || expected.info.Mode() != info.Mode() || !expected.info.ModTime().Equal(info.ModTime()) {
		return fmt.Errorf("collection journal %q metadata changed after validation", path)
	}
	return nil
}

func validateCollectionJournalSummary(path string, actual collectionJournalFileSummary, expected *collectionJournalFileSummary) error {
	if expected == nil {
		return nil
	}
	if actual.recordCount != expected.recordCount ||
		actual.firstLSN != expected.firstLSN ||
		actual.lastLSN != expected.lastLSN ||
		actual.digest != expected.digest {
		return fmt.Errorf("collection journal %q contents changed after validation", path)
	}
	return nil
}

func equalCollectionJournalChecksum(a, b []byte) bool {
	if len(a) != sha256.Size || len(b) != sha256.Size {
		return false
	}
	var diff byte
	for i := range a {
		diff |= a[i] ^ b[i]
	}
	return diff == 0
}

func writeCollectionJournalFrame(writeFn func(*os.File, []byte) (int, error), f *os.File, frame []byte) error {
	for len(frame) > 0 {
		n, err := writeFn(f, frame)
		if n < 0 || n > len(frame) {
			return fmt.Errorf("invalid collection journal write count %d", n)
		}
		frame = frame[n:]
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func syncCollectionJournalDir(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	syncErr := dir.Sync()
	closeErr := dir.Close()
	return errors.Join(syncErr, closeErr)
}
