// Package replication carries a leader's durable journal to a read replica
// over HTTP.
//
// It is transport only. Every durability decision -- what an LSN means, which
// record a replica may accept next, what a snapshot header promises -- already
// belongs to internal/collection; this package moves those bytes and refuses
// to move anything it cannot prove came from the expected store.
//
// The node protocol is versioned independently of the client V3 contract. A
// rolling upgrade negotiates THIS version; the two surfaces have different
// audiences and different compatibility windows, so tying them together would
// mean a client-visible bump every time the follower framing changes.
package replication

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// Version is the node protocol version. Bump it only for a framing change a
// previous follower cannot parse.
const Version = 1

// magic makes a follower pointed at the wrong URL fail on the first five bytes
// instead of trying to parse a proxy error page as journal records.
var magic = [4]byte{'D', 'D', 'R', 'P'}

// MaxRecordBytes bounds a single framed payload. It matches the journal's own
// payload ceiling: a frame larger than the leader could ever have written is a
// corrupt length field, and reading it would let a peer name an allocation.
const MaxRecordBytes = 16 << 20

// Control frame codes. A control frame carries LSN 0, which no real record
// ever has, so it is unambiguous mid-stream. It exists because the leader can
// only discover a gap AFTER the 200 and the preamble are already on the wire,
// and a follower must be able to tell "leader says resync" from "connection
// died".
const (
	// ControlGap: the records the follower asked for are no longer retained.
	// The follower must re-bootstrap from a snapshot.
	ControlGap = 1
	// ControlStoreMismatch: the cursor was issued by a different store.
	ControlStoreMismatch = 2
	// ControlFault: the leader cannot continue serving this stream.
	ControlFault = 3
)

var (
	// ErrBadStream means the bytes are not this protocol at all.
	ErrBadStream = errors.New("replication stream is not a DeepData journal stream")
	// ErrVersionMismatch means the peer speaks a node protocol this build cannot parse.
	ErrVersionMismatch = errors.New("replication stream version is not supported")
	// ErrCorruptFrame means a frame's checksum did not cover its bytes.
	ErrCorruptFrame = errors.New("replication frame failed its checksum")
)

// ControlError is a leader-signalled end of stream. Code is one of the
// Control* constants; the follower branches on it rather than on message text.
type ControlError struct {
	Code    byte
	Message string
}

func (e *ControlError) Error() string {
	name := map[byte]string{
		ControlGap:           "journal gap",
		ControlStoreMismatch: "store mismatch",
		ControlFault:         "leader fault",
	}[e.Code]
	if name == "" {
		name = fmt.Sprintf("control code %d", e.Code)
	}
	return "leader ended replication stream: " + name + ": " + e.Message
}

// Preamble identifies the leader and how far its journal had been written when
// the stream opened. A follower checks StoreID before applying a single record.
type Preamble struct {
	Version   byte
	StoreID   [16]byte
	LatestLSN uint64
}

// crcTable is Castagnoli. It is deliberately NOT the journal's own check --
// the journal frames carry SHA-256, which is the durability check and stays
// authoritative on both sides. This one only has to catch a transport error
// that TCP's 16-bit checksum let through, before a corrupt payload reaches
// ApplyReplicated and gets appended verbatim. CRC32C is the cheap, correct
// tool for that job; re-hashing every record with SHA-256 on the wire would
// cost the leader real throughput to re-answer a question the disk already
// answers.
var crcTable = crc32.MakeTable(crc32.Castagnoli)

// WritePreamble emits the stream header.
func WritePreamble(w io.Writer, p Preamble) error {
	var buf [4 + 1 + 16 + 8]byte
	copy(buf[0:4], magic[:])
	buf[4] = p.Version
	copy(buf[5:21], p.StoreID[:])
	binary.BigEndian.PutUint64(buf[21:29], p.LatestLSN)
	_, err := w.Write(buf[:])
	return err
}

// ReadPreamble parses the stream header and rejects anything that is not this
// protocol at this version.
func ReadPreamble(r io.Reader) (Preamble, error) {
	var buf [4 + 1 + 16 + 8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			return Preamble{}, fmt.Errorf("%w: stream ended inside the header", ErrBadStream)
		}
		return Preamble{}, err
	}
	if [4]byte(buf[0:4]) != magic {
		return Preamble{}, fmt.Errorf("%w: got magic %q", ErrBadStream, buf[0:4])
	}
	p := Preamble{Version: buf[4], LatestLSN: binary.BigEndian.Uint64(buf[21:29])}
	copy(p.StoreID[:], buf[5:21])
	if p.Version != Version {
		return Preamble{}, fmt.Errorf("%w: peer speaks version %d, this build speaks %d", ErrVersionMismatch, p.Version, Version)
	}
	return p, nil
}

// WriteRecord frames one journal record. Payload is written verbatim: the
// replica appends exactly these bytes, so any transformation here would make
// its journal something other than a record-for-record copy.
func WriteRecord(w io.Writer, lsn uint64, payload []byte) error {
	if lsn == 0 {
		return errors.New("replication record LSN cannot be zero")
	}
	return writeFrame(w, lsn, payload)
}

// WriteControl ends the stream in band with a reason the follower can branch on.
func WriteControl(w io.Writer, code byte, message string) error {
	return writeFrame(w, 0, append([]byte{code}, message...))
}

func writeFrame(w io.Writer, lsn uint64, payload []byte) error {
	if len(payload) > MaxRecordBytes {
		return fmt.Errorf("replication payload is %d bytes; maximum is %d", len(payload), MaxRecordBytes)
	}
	var head [12]byte
	binary.BigEndian.PutUint64(head[0:8], lsn)
	binary.BigEndian.PutUint32(head[8:12], uint32(len(payload)))
	sum := crc32.Update(crc32.Checksum(head[:], crcTable), crcTable, payload)
	if _, err := w.Write(head[:]); err != nil {
		return err
	}
	if _, err := w.Write(payload); err != nil {
		return err
	}
	var tail [4]byte
	binary.BigEndian.PutUint32(tail[:], sum)
	_, err := w.Write(tail[:])
	return err
}

// ReadRecord returns the next record's LSN and payload.
//
// The payload aliases buf when it fits, so a caller that retains it across
// calls must copy -- the same contract collection.JournalRecord has, kept
// identical so a follower cannot learn one rule and violate the other.
//
// A clean EOF exactly at a frame boundary is the end of a finite stream and is
// reported as io.EOF. An EOF anywhere inside a frame is a truncated transfer
// and is reported as io.ErrUnexpectedEOF, never as a clean end.
func ReadRecord(r io.Reader, buf *[]byte) (uint64, []byte, error) {
	var head [12]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return 0, nil, err // io.EOF here is a clean end of stream.
	}
	lsn := binary.BigEndian.Uint64(head[0:8])
	length := binary.BigEndian.Uint32(head[8:12])
	if length > MaxRecordBytes {
		return 0, nil, fmt.Errorf("%w: frame claims %d bytes; maximum is %d", ErrCorruptFrame, length, MaxRecordBytes)
	}
	if uint32(cap(*buf)) < length {
		*buf = make([]byte, length)
	}
	payload := (*buf)[:length]
	if _, err := io.ReadFull(r, payload); err != nil {
		return 0, nil, unexpected(err)
	}
	var tail [4]byte
	if _, err := io.ReadFull(r, tail[:]); err != nil {
		return 0, nil, unexpected(err)
	}
	want := crc32.Update(crc32.Checksum(head[:], crcTable), crcTable, payload)
	if got := binary.BigEndian.Uint32(tail[:]); got != want {
		return 0, nil, fmt.Errorf("%w: frame at LSN %d has checksum %08x, computed %08x", ErrCorruptFrame, lsn, got, want)
	}
	if lsn == 0 {
		if len(payload) == 0 {
			return 0, nil, fmt.Errorf("%w: control frame carries no code", ErrCorruptFrame)
		}
		return 0, nil, &ControlError{Code: payload[0], Message: string(payload[1:])}
	}
	return lsn, payload, nil
}

// unexpected converts the EOF that ends a frame's own bytes into the error
// that says so. io.ReadFull already does this for a partial read, but a frame
// whose body is entirely absent returns a bare io.EOF, which a caller would
// otherwise read as a clean end of stream and treat a truncated transfer as
// success.
func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
