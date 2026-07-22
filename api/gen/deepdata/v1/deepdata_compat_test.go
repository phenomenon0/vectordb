package deepdatav1_test

import (
	"crypto/sha256"
	"fmt"
	"testing"

	deepdatav1 "github.com/phenomenon0/vectordb/api/gen/deepdata/v1"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
)

// This hash covers the complete historical V1 file descriptor: every message,
// field number/type, reservation, and service method. Tenant-aware changes
// belong in V3; changing this value requires an explicit compatibility review.
func TestHistoricalV1DescriptorHashIsFrozen(t *testing.T) {
	descriptor := protodesc.ToFileDescriptorProto(deepdatav1.File_deepdata_v1_deepdata_proto)
	encoded, err := proto.MarshalOptions{Deterministic: true}.Marshal(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprintf("%x", sha256.Sum256(encoded))
	const want = "ddc71963d674bebfc3f99dd8ec7b8099c7b95b6c1d0b02ec4c0ce4cd084adb44"
	if got != want {
		t.Fatalf("deepdata.v1 descriptor hash = %s, want %s", got, want)
	}
}
