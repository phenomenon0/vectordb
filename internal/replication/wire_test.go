package replication

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

func leaderID() [16]byte {
	var id [16]byte
	copy(id[:], "leader-store-id!")
	return id
}

// A stream survives the round trip with its LSNs and payload bytes intact.
// Verbatim payload is the property replication rests on: the replica appends
// these exact bytes, so any transformation would break the record-for-record
// copy a further follower depends on.
func TestRoundTripPreservesEveryPayloadByte(t *testing.T) {
	var out bytes.Buffer
	if err := WritePreamble(&out, Preamble{Version: Version, StoreID: leaderID(), LatestLSN: 3}); err != nil {
		t.Fatal(err)
	}
	want := [][]byte{{}, {0x00, 0xff, 0x7f}, []byte("mutation payload")}
	for i, payload := range want {
		if err := WriteRecord(&out, uint64(i+1), payload); err != nil {
			t.Fatal(err)
		}
	}

	in := bytes.NewReader(out.Bytes())
	pre, err := ReadPreamble(in)
	if err != nil {
		t.Fatal(err)
	}
	if pre.StoreID != leaderID() || pre.LatestLSN != 3 {
		t.Fatalf("preamble = %+v", pre)
	}
	var buf []byte
	for i, expect := range want {
		lsn, payload, err := ReadRecord(in, &buf)
		if err != nil {
			t.Fatalf("record %d: %v", i, err)
		}
		if lsn != uint64(i+1) {
			t.Errorf("record %d LSN = %d, want %d", i, lsn, i+1)
		}
		if !bytes.Equal(payload, expect) {
			t.Errorf("record %d payload = %x, want %x", i, payload, expect)
		}
	}
	if _, _, err := ReadRecord(in, &buf); !errors.Is(err, io.EOF) {
		t.Fatalf("end of a finite stream = %v, want io.EOF", err)
	}
}

// A follower pointed at something that is not a leader must fail on the header
// rather than interpret an HTML error page as journal records.
func TestNonProtocolBytesAreRejectedAtTheHeader(t *testing.T) {
	_, err := ReadPreamble(strings.NewReader("<html>502 Bad Gateway</html>"))
	if !errors.Is(err, ErrBadStream) {
		t.Fatalf("err = %v, want ErrBadStream", err)
	}
}

func TestForeignVersionIsRefusedBeforeAnyRecord(t *testing.T) {
	var out bytes.Buffer
	if err := WritePreamble(&out, Preamble{Version: Version + 1, StoreID: leaderID()}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadPreamble(bytes.NewReader(out.Bytes())); !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("err = %v, want ErrVersionMismatch", err)
	}
}

// A flipped payload bit must not reach ApplyReplicated. Without this check the
// replica would append the corrupt bytes verbatim and only notice when the
// mutation failed to decode -- or not at all, if it decoded to something else.
func TestFlippedPayloadBitFailsTheChecksum(t *testing.T) {
	var out bytes.Buffer
	if err := WriteRecord(&out, 7, []byte("mutation payload")); err != nil {
		t.Fatal(err)
	}
	corrupt := out.Bytes()
	corrupt[12+3] ^= 0x01 // inside the payload, past the 12-byte header
	var buf []byte
	if _, _, err := ReadRecord(bytes.NewReader(corrupt), &buf); !errors.Is(err, ErrCorruptFrame) {
		t.Fatalf("err = %v, want ErrCorruptFrame", err)
	}
}

// A connection that dies mid-frame must never look like the clean end of a
// finite stream: the follower would record a short prefix as complete and
// resume past records it never applied.
func TestTruncatedTransferIsNotACleanEnd(t *testing.T) {
	var out bytes.Buffer
	if err := WriteRecord(&out, 7, []byte("mutation payload")); err != nil {
		t.Fatal(err)
	}
	full := out.Bytes()
	for _, cut := range []int{12, 12 + 4, len(full) - 1} {
		var buf []byte
		_, _, err := ReadRecord(bytes.NewReader(full[:cut]), &buf)
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("cut at %d: err = %v, want io.ErrUnexpectedEOF", cut, err)
		}
	}
}

// The leader can only discover a gap after the 200 is already sent, so the
// reason has to travel in band and stay machine-readable.
func TestControlFrameEndsTheStreamWithABranchableCode(t *testing.T) {
	var out bytes.Buffer
	if err := WriteControl(&out, ControlGap, "checkpoint removed LSN 5"); err != nil {
		t.Fatal(err)
	}
	var buf []byte
	_, _, err := ReadRecord(bytes.NewReader(out.Bytes()), &buf)
	var ctl *ControlError
	if !errors.As(err, &ctl) {
		t.Fatalf("err = %v, want *ControlError", err)
	}
	if ctl.Code != ControlGap {
		t.Errorf("code = %d, want ControlGap", ctl.Code)
	}
	if !strings.Contains(ctl.Error(), "checkpoint removed LSN 5") {
		t.Errorf("message lost: %q", ctl.Error())
	}
}

// LSN 0 is the control-frame marker, so a record must never be able to claim it.
func TestRecordCannotClaimTheControlLSN(t *testing.T) {
	if err := WriteRecord(io.Discard, 0, []byte("x")); err == nil {
		t.Fatal("WriteRecord(lsn=0) succeeded; it would be read back as a control frame")
	}
}

// A peer must not be able to name an allocation through the length field.
func TestOversizedLengthIsRefusedWithoutAllocating(t *testing.T) {
	frame := make([]byte, 12)
	frame[8], frame[9], frame[10], frame[11] = 0xff, 0xff, 0xff, 0xff // ~4 GiB
	frame[7] = 1                                                      // LSN 1
	var buf []byte
	_, _, err := ReadRecord(bytes.NewReader(frame), &buf)
	if !errors.Is(err, ErrCorruptFrame) {
		t.Fatalf("err = %v, want ErrCorruptFrame", err)
	}
	if cap(buf) > MaxRecordBytes {
		t.Fatalf("allocated %d bytes from an untrusted length field", cap(buf))
	}
}

// Version 2 added Epoch and EpochStartLSN to the preamble; a follower on this
// build must see both survive the wire, not just the fields version 1 had.
func TestPreambleV2RoundTrip(t *testing.T) {
	var out bytes.Buffer
	want := Preamble{Version: Version, StoreID: leaderID(), LatestLSN: 9, Epoch: 3, EpochStartLSN: 7}
	if err := WritePreamble(&out, want); err != nil {
		t.Fatal(err)
	}
	if got := out.Len(); got != 45 {
		t.Fatalf("preamble is %d bytes, want 45", got)
	}
	got, err := ReadPreamble(bytes.NewReader(out.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("preamble round trip = %+v, want %+v", got, want)
	}
}
