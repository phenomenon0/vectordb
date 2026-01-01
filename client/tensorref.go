package client

import (
	"context"
	"crypto/sha256"
	"fmt"

	"github.com/phenomenon0/Agent-GO/sjson"
)

// TensorRefBridge maps SJSON TensorRef ↔ VectorDB.
// StoreID (0-255) maps to collection name.
// Key is the vector ID (content hash or explicit).
type TensorRefBridge struct {
	client        *Client
	collectionMap map[uint8]string
	defaultStore  uint8
}

// NewTensorRefBridge creates a bridge with VectorDB client.
func NewTensorRefBridge(c *Client) *TensorRefBridge {
	return &TensorRefBridge{
		client:        c,
		collectionMap: make(map[uint8]string),
		defaultStore:  0,
	}
}

// MapStore associates a StoreID with a collection name.
func (b *TensorRefBridge) MapStore(storeID uint8, collection string) {
	b.collectionMap[storeID] = collection
}

// SetDefault sets the default StoreID.
func (b *TensorRefBridge) SetDefault(storeID uint8) {
	b.defaultStore = storeID
}

// Store inserts doc and returns TensorRef. Empty id = content hash.
func (b *TensorRefBridge) Store(ctx context.Context, doc string, meta map[string]string, id string) (*sjson.Value, error) {
	if id == "" {
		hash := sha256.Sum256([]byte(doc))
		id = fmt.Sprintf("%x", hash[:16])
	}

	req := InsertRequest{
		ID:         id,
		Doc:        doc,
		Meta:       meta,
		Collection: b.collectionMap[b.defaultStore],
		Upsert:     true,
	}

	resp, err := b.client.Insert(ctx, req)
	if err != nil {
		return nil, err
	}

	return sjson.TensorRef(b.defaultStore, []byte(resp.ID)), nil
}

// Search returns TensorRefs sorted by similarity.
func (b *TensorRefBridge) Search(ctx context.Context, query string, topK int) ([]*sjson.Value, []float32, error) {
	req := QueryRequest{
		Query:      query,
		TopK:       topK,
		Collection: b.collectionMap[b.defaultStore],
	}

	resp, err := b.client.Query(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	refs := make([]*sjson.Value, len(resp.IDs))
	for i, id := range resp.IDs {
		refs[i] = sjson.TensorRef(b.defaultStore, []byte(id))
	}

	return refs, resp.Scores, nil
}

// Resolve looks up TensorRef and returns (doc, meta, error).
func (b *TensorRefBridge) Resolve(ctx context.Context, ref sjson.TensorRefData) (string, map[string]string, error) {
	// Use ID as query to find the document
	req := QueryRequest{
		Query:       string(ref.Key),
		TopK:        1,
		Collection:  b.collectionMap[ref.StoreID],
		IncludeMeta: true,
	}

	resp, err := b.client.Query(ctx, req)
	if err != nil {
		return "", nil, err
	}

	if len(resp.Docs) == 0 {
		return "", nil, fmt.Errorf("not found: store=%d key=%s", ref.StoreID, ref.Key)
	}

	var meta map[string]string
	if len(resp.Meta) > 0 {
		meta = resp.Meta[0]
	}

	return resp.Docs[0], meta, nil
}

// BatchStore inserts multiple docs.
func (b *TensorRefBridge) BatchStore(ctx context.Context, docs []string, metas []map[string]string) ([]*sjson.Value, error) {
	batchDocs := make([]BatchDoc, len(docs))
	for i, doc := range docs {
		hash := sha256.Sum256([]byte(doc))
		var meta map[string]string
		if i < len(metas) {
			meta = metas[i]
		}
		batchDocs[i] = BatchDoc{
			ID:         fmt.Sprintf("%x", hash[:16]),
			Doc:        doc,
			Meta:       meta,
			Collection: b.collectionMap[b.defaultStore],
		}
	}

	resp, err := b.client.BatchInsert(ctx, BatchInsertRequest{Docs: batchDocs, Upsert: true})
	if err != nil {
		return nil, err
	}

	refs := make([]*sjson.Value, len(resp.IDs))
	for i, id := range resp.IDs {
		refs[i] = sjson.TensorRef(b.defaultStore, []byte(id))
	}
	return refs, nil
}

// Delete removes vector by TensorRef.
func (b *TensorRefBridge) Delete(ctx context.Context, ref sjson.TensorRefData) error {
	_, err := b.client.Delete(ctx, DeleteRequest{ID: string(ref.Key)})
	return err
}
