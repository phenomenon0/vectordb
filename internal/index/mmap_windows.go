//go:build windows

package index

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

func mmapCreate(fd int, size int) ([]byte, error) {
	handle := windows.Handle(uintptr(fd))

	// Create file mapping
	mapHandle, err := windows.CreateFileMapping(handle, nil, windows.PAGE_READWRITE, 0, uint32(size), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create file mapping: %w", err)
	}

	// Map view of file
	addr, err := windows.MapViewOfFile(mapHandle, windows.FILE_MAP_READ|windows.FILE_MAP_WRITE, 0, 0, uintptr(size))
	if err != nil {
		windows.CloseHandle(mapHandle)
		return nil, fmt.Errorf("failed to map view: %w", err)
	}

	// We intentionally leak mapHandle — it stays alive until the view is unmapped.
	// Store it if cleanup is needed, but for our usage the process lifetime is fine.
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	return data, nil
}

func mmapUnmap(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&data[0])))
}

func mmapSync(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return windows.FlushViewOfFile(uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)))
}

// fdFromFile extracts the Windows handle as an int for mmapCreate.
// On Windows, os.File.Fd() returns a handle, not a Unix fd.
func fdFromFile(f *os.File) int {
	return int(f.Fd())
}
