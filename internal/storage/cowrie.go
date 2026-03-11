package storage

import (
	"fmt"
	"io"
	"time"

	cowrie "github.com/Neumenon/cowrie/go"
	"github.com/Neumenon/cowrie/go/ucodec"
	"github.com/phenomenon0/vectordb/internal/cowrieutil"
)

// CowrieFormat implements Format using Cowrie binary encoding.
// Optimizes embedding storage using native tensor type (~48% smaller for float32 arrays).
type CowrieFormat struct {
	// UseCompression enables Zstd compression (additional ~40% reduction)
	UseCompression bool

	// UseDeltaPrediction enables delta-predictive encoding for embeddings.
	// When true, correlated embedding data is delta-encoded before zstd compression,
	// achieving 40-55% smaller than raw float32 (vs 10-15% with zstd alone).
	// Falls back to normal tensor encoding if data correlation is too low.
	UseDeltaPrediction bool
}

func init() {
	Register(&CowrieFormat{UseCompression: false})
	Register(&CowrieFormat{UseCompression: true})
	Register(&CowrieFormat{UseCompression: true, UseDeltaPrediction: true})
}

func (s *CowrieFormat) Name() string {
	if s.UseCompression && s.UseDeltaPrediction {
		return "cowrie-delta-zstd"
	}
	if s.UseCompression {
		return "cowrie-zstd"
	}
	return "cowrie"
}

func (s *CowrieFormat) Extension() string {
	if s.UseCompression && s.UseDeltaPrediction {
		return ".cowrie.delta.zst"
	}
	if s.UseCompression {
		return ".cowrie.zst"
	}
	return ".cowrie"
}

func (s *CowrieFormat) Save(w io.Writer, p *Payload) error {
	obj := cowrie.Object()

	// Store embeddings - choose encoding strategy
	deltaEncoded := false
	if len(p.Data) > 0 {
		if s.UseDeltaPrediction && ucodec.ShouldUseDelta(p.Data, 0.5) {
			deltaBytes, err := ucodec.EncodeDelta(p.Data)
			if err == nil {
				obj.Set("data_delta", cowrie.Bytes(deltaBytes))
				deltaEncoded = true
			}
		}
		if !deltaEncoded {
			obj.Set("data", cowrieutil.EncodeFloat32Tensor(p.Data))
		}
	}
	obj.Set("delta_encoded", cowrie.Bool(deltaEncoded))

	// Scalar fields
	obj.Set("dim", cowrie.Int64(int64(p.Dim)))
	obj.Set("next", cowrie.Int64(p.Next))
	obj.Set("count", cowrie.Int64(int64(p.Count)))
	obj.Set("sum_doc_l", cowrie.Int64(int64(p.SumDocL)))
	obj.Set("checksum", cowrie.String(p.Checksum))
	obj.Set("last_saved", cowrie.String(p.LastSaved.Format(time.RFC3339Nano)))

	// String arrays
	obj.Set("docs", cowrieutil.EncodeStringArray(p.Docs))
	obj.Set("ids", cowrieutil.EncodeStringArray(p.IDs))

	// HNSW binary blob
	if len(p.HNSW) > 0 {
		obj.Set("hnsw", cowrie.Bytes(p.HNSW))
	}

	// Maps
	obj.Set("meta", cowrieutil.EncodeStringMapMap(p.Meta))
	obj.Set("deleted", cowrieutil.EncodeBoolMap(p.Deleted))
	obj.Set("coll", cowrieutil.EncodeStringMapUint64(p.Coll))
	obj.Set("lex_tf", cowrieutil.EncodeIntMapMap(p.LexTF))
	obj.Set("doc_len", cowrieutil.EncodeIntMapUint64(p.DocLen))
	obj.Set("df", cowrieutil.EncodeStringIntMap(p.DF))
	obj.Set("num_meta", cowrieutil.EncodeFloat64MapMap(p.NumMeta))
	obj.Set("time_meta", cowrieutil.EncodeTimeMapMap(p.TimeMeta))

	// Encode to bytes
	var data []byte
	var err error

	if s.UseCompression {
		data, err = cowrie.EncodeFramed(obj, cowrie.CompressionZstd)
	} else {
		data, err = cowrie.Encode(obj)
	}

	if err != nil {
		return err
	}

	_, err = w.Write(data)
	return err
}

func (s *CowrieFormat) Load(r io.Reader) (*Payload, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var obj *cowrie.Value
	if s.UseCompression {
		obj, err = cowrie.DecodeFramed(data)
	} else {
		obj, err = cowrie.Decode(data)
	}
	if err != nil {
		return nil, err
	}

	p := &Payload{}

	// Check if delta encoding was used
	isDelta := false
	if v := obj.Get("delta_encoded"); v != nil {
		isDelta = cowrieutil.SafeBool(v)
	}

	// Decode embeddings
	if isDelta {
		if v := obj.Get("data_delta"); v != nil && v.Type() == cowrie.TypeBytes {
			decoded, err := ucodec.DecodeDelta(v.Bytes())
			if err != nil {
				return nil, fmt.Errorf("delta decode failed: %w", err)
			}
			p.Data = decoded
		}
	} else {
		if dataVal := obj.Get("data"); dataVal != nil {
			p.Data = cowrieutil.DecodeFloat32Tensor(dataVal)
		}
	}

	// Scalar fields
	if v := obj.Get("dim"); v != nil {
		p.Dim = int(cowrieutil.SafeInt64(v))
	}
	if v := obj.Get("next"); v != nil {
		p.Next = cowrieutil.SafeInt64(v)
	}
	if v := obj.Get("count"); v != nil {
		p.Count = int(cowrieutil.SafeInt64(v))
	}
	if v := obj.Get("sum_doc_l"); v != nil {
		p.SumDocL = int(cowrieutil.SafeInt64(v))
	}
	if v := obj.Get("checksum"); v != nil {
		p.Checksum = cowrieutil.SafeString(v)
	}
	if v := obj.Get("last_saved"); v != nil {
		if t, err := time.Parse(time.RFC3339Nano, cowrieutil.SafeString(v)); err == nil {
			p.LastSaved = t
		}
	}

	// String arrays
	if v := obj.Get("docs"); v != nil {
		p.Docs = cowrieutil.DecodeStringArray(v)
	}
	if v := obj.Get("ids"); v != nil {
		p.IDs = cowrieutil.DecodeStringArray(v)
	}

	// HNSW binary
	if v := obj.Get("hnsw"); v != nil && v.Type() == cowrie.TypeBytes {
		p.HNSW = v.Bytes()
	}

	// Maps
	if v := obj.Get("meta"); v != nil {
		p.Meta = cowrieutil.DecodeStringMapMap(v)
	}
	if v := obj.Get("deleted"); v != nil {
		p.Deleted = cowrieutil.DecodeBoolMap(v)
	}
	if v := obj.Get("coll"); v != nil {
		p.Coll = cowrieutil.DecodeStringMapUint64(v)
	}
	if v := obj.Get("lex_tf"); v != nil {
		p.LexTF = cowrieutil.DecodeIntMapMap(v)
	}
	if v := obj.Get("doc_len"); v != nil {
		p.DocLen = cowrieutil.DecodeIntMapUint64(v)
	}
	if v := obj.Get("df"); v != nil {
		p.DF = cowrieutil.DecodeStringIntMap(v)
	}
	if v := obj.Get("num_meta"); v != nil {
		p.NumMeta = cowrieutil.DecodeFloat64MapMap(v)
	}
	if v := obj.Get("time_meta"); v != nil {
		p.TimeMeta = cowrieutil.DecodeTimeMapMap(v)
	}

	return p, nil
}
