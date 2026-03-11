package wal

import (
	"fmt"

	cowrie "github.com/Neumenon/cowrie/go"
	"github.com/phenomenon0/vectordb/internal/cowrieutil"
)

// encodeEntryCowrie serializes a WAL entry using cowrie binary format.
// Vectors are encoded as native tensors (4 bytes/float vs ~10 bytes/float in JSON).
func encodeEntryCowrie(entry *Entry) ([]byte, error) {
	obj := cowrie.Object()
	obj.Set("lsn", cowrie.Int64(int64(entry.LSN)))
	obj.Set("op", cowrie.Int64(int64(entry.Op)))
	obj.Set("ts", cowrie.Int64(entry.Timestamp))

	if entry.Collection != "" {
		obj.Set("collection", cowrie.String(entry.Collection))
	}
	if entry.ID != "" {
		obj.Set("id", cowrie.String(entry.ID))
	}
	if len(entry.Vector) > 0 {
		obj.Set("vector", cowrieutil.EncodeFloat32Tensor(entry.Vector))
	}
	if entry.Doc != "" {
		obj.Set("doc", cowrie.String(entry.Doc))
	}
	if len(entry.Meta) > 0 {
		obj.Set("meta", cowrieutil.EncodeStringStringMap(entry.Meta))
	}
	if entry.TenantID != "" {
		obj.Set("tenant_id", cowrie.String(entry.TenantID))
	}
	if len(entry.Batch) > 0 {
		items := make([]*cowrie.Value, len(entry.Batch))
		for i, b := range entry.Batch {
			item := cowrie.Object()
			item.Set("id", cowrie.String(b.ID))
			if len(b.Vector) > 0 {
				item.Set("vector", cowrieutil.EncodeFloat32Tensor(b.Vector))
			}
			if b.Doc != "" {
				item.Set("doc", cowrie.String(b.Doc))
			}
			if len(b.Meta) > 0 {
				item.Set("meta", cowrieutil.EncodeStringStringMap(b.Meta))
			}
			items[i] = item
		}
		obj.Set("batch", cowrie.Array(items...))
	}

	return cowrie.Encode(obj)
}

// decodeEntryCowrie deserializes a WAL entry from cowrie binary format.
func decodeEntryCowrie(data []byte) (*Entry, error) {
	obj, err := cowrie.Decode(data)
	if err != nil {
		return nil, fmt.Errorf("cowrie decode failed: %w", err)
	}

	entry := &Entry{
		LSN:       cowrieutil.SafeUint64(obj.Get("lsn")),
		Op:        OpType(cowrieutil.SafeInt64(obj.Get("op"))),
		Timestamp: cowrieutil.SafeInt64(obj.Get("ts")),
	}

	if v := obj.Get("collection"); v != nil {
		entry.Collection = cowrieutil.SafeString(v)
	}
	if v := obj.Get("id"); v != nil {
		entry.ID = cowrieutil.SafeString(v)
	}
	if v := obj.Get("vector"); v != nil {
		entry.Vector = cowrieutil.DecodeFloat32Tensor(v)
	}
	if v := obj.Get("doc"); v != nil {
		entry.Doc = cowrieutil.SafeString(v)
	}
	if v := obj.Get("meta"); v != nil {
		entry.Meta = cowrieutil.DecodeStringStringMap(v)
	}
	if v := obj.Get("tenant_id"); v != nil {
		entry.TenantID = cowrieutil.SafeString(v)
	}
	if v := obj.Get("batch"); v != nil && v.Type() == cowrie.TypeArray {
		arr := v.Array()
		entry.Batch = make([]BatchEntry, len(arr))
		for i, elem := range arr {
			b := BatchEntry{}
			if iv := elem.Get("id"); iv != nil {
				b.ID = cowrieutil.SafeString(iv)
			}
			if iv := elem.Get("vector"); iv != nil {
				b.Vector = cowrieutil.DecodeFloat32Tensor(iv)
			}
			if iv := elem.Get("doc"); iv != nil {
				b.Doc = cowrieutil.SafeString(iv)
			}
			if iv := elem.Get("meta"); iv != nil {
				b.Meta = cowrieutil.DecodeStringStringMap(iv)
			}
			entry.Batch[i] = b
		}
	}

	return entry, nil
}

// isJSONPayload returns true if the data starts with '{', indicating JSON encoding.
// Used for backward compatibility: old WAL entries are JSON, new ones are cowrie.
func isJSONPayload(data []byte) bool {
	return len(data) > 0 && data[0] == '{'
}
