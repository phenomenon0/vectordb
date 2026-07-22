//go:build linux

package collection

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

type collectionStoreLock struct {
	file *os.File
}

func acquireCollectionStoreLock(path string) (*collectionStoreLock, error) {
	fd, err := syscall.Open(path, syscall.O_CREAT|syscall.O_RDWR|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open collection store lock: %w", err)
	}
	f := os.NewFile(uintptr(fd), path)
	if f == nil {
		_ = syscall.Close(fd)
		return nil, errors.New("open collection store lock: invalid file descriptor")
	}
	closeWith := func(cause error) (*collectionStoreLock, error) {
		return nil, errors.Join(cause, f.Close())
	}
	info, err := f.Stat()
	if err != nil {
		return closeWith(fmt.Errorf("stat collection store lock: %w", err))
	}
	if !info.Mode().IsRegular() {
		return closeWith(errors.New("collection store lock is not a regular file"))
	}
	if err := f.Chmod(0o600); err != nil {
		return closeWith(fmt.Errorf("secure collection store lock: %w", err))
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return closeWith(fmt.Errorf("collection store is already open: %w", err))
		}
		return closeWith(fmt.Errorf("lock collection store: %w", err))
	}
	if err := f.Sync(); err != nil {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		return closeWith(fmt.Errorf("sync collection store lock: %w", err))
	}
	if err := syncCollectionJournalDir(filepath.Dir(path)); err != nil {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		return closeWith(fmt.Errorf("sync collection store lock directory: %w", err))
	}
	return &collectionStoreLock{file: f}, nil
}

func (l *collectionStoreLock) release() error {
	if l == nil || l.file == nil {
		return nil
	}
	f := l.file
	l.file = nil
	return errors.Join(syscall.Flock(int(f.Fd()), syscall.LOCK_UN), f.Close())
}
