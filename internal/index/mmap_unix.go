//go:build !windows

package index

import (
	"fmt"
	"syscall"

	"golang.org/x/sys/unix"
)

func mmapCreate(fd int, size int) ([]byte, error) {
	data, err := syscall.Mmap(fd, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("failed to mmap file: %w", err)
	}
	// Best-effort hint for random access pattern
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return data, nil
}

func mmapUnmap(data []byte) error {
	return syscall.Munmap(data)
}

func mmapSync(data []byte) error {
	return unix.Msync(data, unix.MS_SYNC)
}
