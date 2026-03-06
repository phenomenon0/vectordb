package main

import (
	"context"
	"fmt"
	"io"

	"github.com/phenomenon0/vectordb/wal"
)

// WALAdapter bridges the VectorStore with the new WAL package.
// It provides durability guarantees with CRC checksums, segment rotation,
// and efficient recovery.
type WALAdapter struct {
	wal    *wal.WAL
	config wal.Config
}

// NewWALAdapter creates a new WAL adapter for the VectorStore.
func NewWALAdapter(dir string) (*WALAdapter, error) {
	config := wal.DefaultConfig(dir)
	w, err := wal.Open(config)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	return &WALAdapter{
		wal:    w,
		config: config,
	}, nil
}

// NewWALAdapterWithConfig creates a WAL adapter with custom configuration.
func NewWALAdapterWithConfig(config wal.Config) (*WALAdapter, error) {
	w, err := wal.Open(config)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL: %w", err)
	}

	return &WALAdapter{
		wal:    w,
		config: config,
	}, nil
}

// Insert logs an insert operation to the WAL.
func (wa *WALAdapter) Insert(id, doc, collection, tenantID string, vector []float32, meta map[string]string) error {
	entry := &wal.Entry{
		Op:         wal.OpInsert,
		ID:         id,
		Doc:        doc,
		Collection: collection,
		TenantID:   tenantID,
		Vector:     vector,
		Meta:       meta,
	}
	return wa.wal.Write(entry)
}

// Update logs an update operation to the WAL.
func (wa *WALAdapter) Update(id, doc, collection, tenantID string, vector []float32, meta map[string]string) error {
	entry := &wal.Entry{
		Op:         wal.OpUpdate,
		ID:         id,
		Doc:        doc,
		Collection: collection,
		TenantID:   tenantID,
		Vector:     vector,
		Meta:       meta,
	}
	return wa.wal.Write(entry)
}

// Delete logs a delete operation to the WAL.
func (wa *WALAdapter) Delete(id, collection, tenantID string) error {
	entry := &wal.Entry{
		Op:         wal.OpDelete,
		ID:         id,
		Collection: collection,
		TenantID:   tenantID,
	}
	return wa.wal.Write(entry)
}

// BatchInsert logs a batch insert operation to the WAL.
func (wa *WALAdapter) BatchInsert(collection, tenantID string, items []BatchItem) error {
	batch := make([]wal.BatchEntry, len(items))
	for i, item := range items {
		batch[i] = wal.BatchEntry{
			ID:     item.ID,
			Vector: item.Vector,
			Doc:    item.Doc,
			Meta:   item.Meta,
		}
	}

	entry := &wal.Entry{
		Op:         wal.OpBatchInsert,
		Collection: collection,
		TenantID:   tenantID,
		Batch:      batch,
	}
	return wa.wal.Write(entry)
}

// BatchItem represents an item in a batch operation.
type BatchItem struct {
	ID     string
	Vector []float32
	Doc    string
	Meta   map[string]string
}

// CreateCollection logs a collection creation to the WAL.
func (wa *WALAdapter) CreateCollection(collection string, tenantID string) error {
	entry := &wal.Entry{
		Op:         wal.OpCreateCollection,
		Collection: collection,
		TenantID:   tenantID,
	}
	return wa.wal.Write(entry)
}

// DeleteCollection logs a collection deletion to the WAL.
func (wa *WALAdapter) DeleteCollection(collection string, tenantID string) error {
	entry := &wal.Entry{
		Op:         wal.OpDeleteCollection,
		Collection: collection,
		TenantID:   tenantID,
	}
	return wa.wal.Write(entry)
}

// Sync forces a sync of the WAL to disk.
func (wa *WALAdapter) Sync() error {
	return wa.wal.Sync()
}

// Close closes the WAL.
func (wa *WALAdapter) Close() error {
	return wa.wal.Close()
}

// LSN returns the current Log Sequence Number.
func (wa *WALAdapter) LSN() uint64 {
	return wa.wal.LSN()
}

// Truncate removes old WAL entries before the given LSN.
func (wa *WALAdapter) Truncate(beforeLSN uint64) error {
	return wa.wal.Truncate(beforeLSN)
}

// Stats returns WAL statistics.
func (wa *WALAdapter) Stats() (*wal.Stats, error) {
	return wal.GetStats(wa.config.Dir)
}

// WALRecoveryHandler is called for each WAL entry during recovery.
type WALRecoveryHandler interface {
	OnInsert(ctx context.Context, entry *wal.Entry) error
	OnUpdate(ctx context.Context, entry *wal.Entry) error
	OnDelete(ctx context.Context, entry *wal.Entry) error
	OnBatchInsert(ctx context.Context, entry *wal.Entry) error
	OnCreateCollection(ctx context.Context, entry *wal.Entry) error
	OnDeleteCollection(ctx context.Context, entry *wal.Entry) error
}

// Replay replays all WAL entries using the provided handler.
func (wa *WALAdapter) Replay(ctx context.Context, handler WALRecoveryHandler) error {
	return wal.Replay(wa.config.Dir, func(entry *wal.Entry) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		var err error
		switch entry.Op {
		case wal.OpInsert:
			err = handler.OnInsert(ctx, entry)
		case wal.OpUpdate:
			err = handler.OnUpdate(ctx, entry)
		case wal.OpDelete:
			err = handler.OnDelete(ctx, entry)
		case wal.OpBatchInsert:
			err = handler.OnBatchInsert(ctx, entry)
		case wal.OpCreateCollection:
			err = handler.OnCreateCollection(ctx, entry)
		case wal.OpDeleteCollection:
			err = handler.OnDeleteCollection(ctx, entry)
		}

		// Continue on errors (log them)
		if err != nil {
			fmt.Printf("WAL replay error (LSN %d, op %s): %v\n", entry.LSN, entry.Op, err)
		}
		return true
	})
}

// ReplayFrom replays WAL entries starting from a specific LSN.
func (wa *WALAdapter) ReplayFrom(ctx context.Context, fromLSN uint64, handler WALRecoveryHandler) error {
	return wal.ReplayFrom(wa.config.Dir, fromLSN, func(entry *wal.Entry) bool {
		select {
		case <-ctx.Done():
			return false
		default:
		}

		var err error
		switch entry.Op {
		case wal.OpInsert:
			err = handler.OnInsert(ctx, entry)
		case wal.OpUpdate:
			err = handler.OnUpdate(ctx, entry)
		case wal.OpDelete:
			err = handler.OnDelete(ctx, entry)
		case wal.OpBatchInsert:
			err = handler.OnBatchInsert(ctx, entry)
		case wal.OpCreateCollection:
			err = handler.OnCreateCollection(ctx, entry)
		case wal.OpDeleteCollection:
			err = handler.OnDeleteCollection(ctx, entry)
		}

		if err != nil {
			fmt.Printf("WAL replay error (LSN %d, op %s): %v\n", entry.LSN, entry.Op, err)
		}
		return true
	})
}

// VectorStoreRecoveryHandler implements WALRecoveryHandler for VectorStore.
type VectorStoreRecoveryHandler struct {
	store *VectorStore
}

// NewVectorStoreRecoveryHandler creates a new recovery handler for VectorStore.
func NewVectorStoreRecoveryHandler(store *VectorStore) *VectorStoreRecoveryHandler {
	return &VectorStoreRecoveryHandler{store: store}
}

// OnInsert handles insert operations during recovery.
func (h *VectorStoreRecoveryHandler) OnInsert(ctx context.Context, entry *wal.Entry) error {
	_, err := h.store.Add(entry.Vector, entry.Doc, entry.ID, entry.Meta, entry.Collection, entry.TenantID)
	return err
}

// OnUpdate handles update operations during recovery.
func (h *VectorStoreRecoveryHandler) OnUpdate(ctx context.Context, entry *wal.Entry) error {
	_, err := h.store.Upsert(entry.Vector, entry.Doc, entry.ID, entry.Meta, entry.Collection, entry.TenantID)
	return err
}

// OnDelete handles delete operations during recovery.
func (h *VectorStoreRecoveryHandler) OnDelete(ctx context.Context, entry *wal.Entry) error {
	h.store.Delete(entry.ID)
	return nil
}

// OnBatchInsert handles batch insert operations during recovery.
func (h *VectorStoreRecoveryHandler) OnBatchInsert(ctx context.Context, entry *wal.Entry) error {
	for _, item := range entry.Batch {
		_, err := h.store.Add(item.Vector, item.Doc, item.ID, item.Meta, entry.Collection, entry.TenantID)
		if err != nil {
			return err
		}
	}
	return nil
}

// OnCreateCollection handles collection creation during recovery.
func (h *VectorStoreRecoveryHandler) OnCreateCollection(ctx context.Context, entry *wal.Entry) error {
	// Collection creation is handled lazily in VectorStore
	return nil
}

// OnDeleteCollection handles collection deletion during recovery.
func (h *VectorStoreRecoveryHandler) OnDeleteCollection(ctx context.Context, entry *wal.Entry) error {
	// Delete all vectors in the collection
	h.store.Lock()
	defer h.store.Unlock()

	for hid, coll := range h.store.Coll {
		if coll == entry.Collection {
			h.store.Deleted[hid] = true
		}
	}
	return nil
}

// ReadAllEntries reads all WAL entries without applying them.
func (wa *WALAdapter) ReadAllEntries() ([]*wal.Entry, error) {
	return wal.ReadAll(wa.config.Dir)
}

// Iterator returns a new iterator over all WAL entries.
func (wa *WALAdapter) Iterator() (*wal.WALIterator, error) {
	return wal.NewIterator(wa.config.Dir)
}

// EntryIterator is a convenience wrapper for iterating WAL entries.
type EntryIterator struct {
	it *wal.WALIterator
}

// NewEntryIterator creates a new entry iterator.
func (wa *WALAdapter) NewEntryIterator() (*EntryIterator, error) {
	it, err := wal.NewIterator(wa.config.Dir)
	if err != nil {
		return nil, err
	}
	return &EntryIterator{it: it}, nil
}

// Next returns the next entry or io.EOF when done.
func (ei *EntryIterator) Next() (*wal.Entry, error) {
	return ei.it.Next()
}

// Close closes the iterator.
func (ei *EntryIterator) Close() error {
	return ei.it.Close()
}

// EnableWAL enables the new WAL system for a VectorStore.
// Call this during store initialization to use the new WAL package.
func EnableWAL(store *VectorStore, walDir string) (*WALAdapter, error) {
	adapter, err := NewWALAdapter(walDir)
	if err != nil {
		return nil, err
	}

	// Set up the WAL hook to forward writes to the new WAL
	store.SetWALHook(func(entry walEntry) {
		var walErr error
		switch entry.Op {
		case "insert":
			walErr = adapter.Insert(entry.ID, entry.Doc, entry.Coll, entry.Tenant, entry.Vec, entry.Meta)
		case "upsert":
			walErr = adapter.Update(entry.ID, entry.Doc, entry.Coll, entry.Tenant, entry.Vec, entry.Meta)
		case "delete":
			walErr = adapter.Delete(entry.ID, entry.Coll, entry.Tenant)
		}
		if walErr != nil {
			fmt.Printf("WAL write error: %v\n", walErr)
		}
	})

	return adapter, nil
}

// RecoverFromWAL recovers a VectorStore from WAL entries.
func RecoverFromWAL(store *VectorStore, walDir string) error {
	adapter, err := NewWALAdapter(walDir)
	if err != nil {
		return fmt.Errorf("failed to open WAL for recovery: %w", err)
	}
	defer adapter.Close()

	handler := NewVectorStoreRecoveryHandler(store)

	// Disable the store's internal WAL during recovery
	origPath := store.walPath
	origHook := store.walHook
	store.walPath = ""
	store.walHook = nil
	defer func() {
		store.walPath = origPath
		store.walHook = origHook
	}()

	ctx := context.Background()
	return adapter.Replay(ctx, handler)
}

// WALStreamAdapter streams WAL entries over HTTP (for replication).
type WALStreamAdapter struct {
	adapter *WALAdapter
	lastLSN uint64
}

// NewWALStreamAdapter creates a streaming adapter.
func NewWALStreamAdapter(adapter *WALAdapter) *WALStreamAdapter {
	return &WALStreamAdapter{
		adapter: adapter,
		lastLSN: 0,
	}
}

// GetEntriesSince returns WAL entries since the given LSN.
func (wsa *WALStreamAdapter) GetEntriesSince(sinceSeq uint64) ([]*wal.Entry, error) {
	it, err := wsa.adapter.Iterator()
	if err != nil {
		return nil, err
	}
	defer it.Close()

	var entries []*wal.Entry
	for {
		entry, err := it.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return entries, err
		}
		if entry.LSN > sinceSeq {
			entries = append(entries, entry)
		}
	}
	return entries, nil
}
