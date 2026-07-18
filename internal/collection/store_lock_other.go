//go:build !linux

package collection

import (
	"errors"
	"fmt"
)

var errPersistentCollectionsUnsupported = errors.New("persistent collection storage is supported only on Linux in this release")

type collectionStoreLock struct{}

func acquireCollectionStoreLock(path string) (*collectionStoreLock, error) {
	return nil, fmt.Errorf("%w: %s", errPersistentCollectionsUnsupported, path)
}

func (l *collectionStoreLock) release() error { return nil }
