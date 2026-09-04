package main

import (
	"net"
	"testing"
)

func unusedLoopbackAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}

func TestBindAPIListenersFailsIfHTTPIsOccupied(t *testing.T) {
	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer occupied.Close()

	httpListener, grpcListener, err := bindAPIListeners(occupied.Addr().String(), "")
	if err == nil {
		t.Fatal("listener bind unexpectedly succeeded")
	}
	if httpListener != nil || grpcListener != nil {
		t.Fatal("failed bind returned a live listener")
	}
}

func TestBindAPIListenersReleasesHTTPIfGRPCBindFails(t *testing.T) {
	grpcOccupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer grpcOccupied.Close()
	httpAddress := unusedLoopbackAddress(t)

	httpListener, grpcListener, err := bindAPIListeners(httpAddress, grpcOccupied.Addr().String())
	if err == nil {
		t.Fatal("complete listener bind unexpectedly succeeded")
	}
	if httpListener != nil || grpcListener != nil {
		t.Fatal("failed complete bind returned a live listener")
	}

	rebound, err := net.Listen("tcp", httpAddress)
	if err != nil {
		t.Fatalf("HTTP listener was not released after gRPC failure: %v", err)
	}
	rebound.Close()
}

func TestBindAPIListenersRejectsSharedPortAndReleasesIt(t *testing.T) {
	address := unusedLoopbackAddress(t)
	httpListener, grpcListener, err := bindAPIListeners(address, address)
	if err == nil {
		t.Fatal("HTTP and gRPC unexpectedly bound the same address")
	}
	if httpListener != nil || grpcListener != nil {
		t.Fatal("shared-port failure returned a live listener")
	}

	rebound, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("shared port remained occupied after startup refusal: %v", err)
	}
	rebound.Close()
}

func TestBindAPIListenersAllowsExplicitGRPCDisable(t *testing.T) {
	httpListener, grpcListener, err := bindAPIListeners("127.0.0.1:0", "")
	if err != nil {
		t.Fatal(err)
	}
	defer httpListener.Close()
	if grpcListener != nil {
		t.Fatal("disabled gRPC unexpectedly returned a listener")
	}
}

func TestCanonicalInsecureDevelopmentAddressesAreLoopbackOnly(t *testing.T) {
	httpAddress, grpcAddress, err := canonicalListenerAddresses(8080, 50051, true, "")
	if err != nil {
		t.Fatal(err)
	}
	if httpAddress != "127.0.0.1:8080" || grpcAddress != "127.0.0.1:50051" {
		t.Fatalf("insecure development addresses = %q and %q", httpAddress, grpcAddress)
	}

	httpAddress, grpcAddress, err = canonicalListenerAddresses(8080, 50051, false, "")
	if err != nil {
		t.Fatal(err)
	}
	if httpAddress != ":8080" || grpcAddress != ":50051" {
		t.Fatalf("authenticated deployment addresses = %q and %q", httpAddress, grpcAddress)
	}

	_, grpcAddress, err = canonicalListenerAddresses(8080, 0, true, "")
	if err != nil {
		t.Fatal(err)
	}
	if grpcAddress != "" {
		t.Fatalf("disabled gRPC address = %q, want empty", grpcAddress)
	}

	httpAddress, grpcAddress, err = canonicalListenerAddresses(8080, 50051, false, "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	if httpAddress != "127.0.0.1:8080" || grpcAddress != "127.0.0.1:50051" {
		t.Fatalf("configured loopback addresses = %q and %q", httpAddress, grpcAddress)
	}

	for _, host := range []string{"localhost", " 127.0.0.1", "0.0.0.0"} {
		if _, _, err := canonicalListenerAddresses(8080, 50051, true, host); err == nil {
			t.Errorf("insecure configured host %q was accepted", host)
		}
	}
}
