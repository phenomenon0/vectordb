package storage

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"time"

	"github.com/phenomenon0/Agent-GO/cowrie"
)

// Safe value extraction helpers - prevent panics from type mismatches

func safeString(v *cowrie.Value) string {
	if v == nil {
		return ""
	}
	switch v.Type() {
	case cowrie.TypeString:
		return v.String()
	case cowrie.TypeInt64:
		return fmt.Sprintf("%d", v.Int64())
	case cowrie.TypeUint64:
		return fmt.Sprintf("%d", v.Uint64())
	case cowrie.TypeFloat64:
		return fmt.Sprintf("%g", v.Float64())
	case cowrie.TypeBool:
		return fmt.Sprintf("%t", v.Bool())
	default:
		return fmt.Sprint(cowrie.ToAny(v))
	}
}

func safeInt64(v *cowrie.Value) int64 {
	if v == nil {
		return 0
	}
	switch v.Type() {
	case cowrie.TypeInt64:
		return v.Int64()
	case cowrie.TypeUint64:
		return int64(v.Uint64())
	case cowrie.TypeFloat64:
		return int64(v.Float64())
	default:
		return 0
	}
}

func safeFloat64(v *cowrie.Value) float64 {
	if v == nil {
		return 0
	}
	switch v.Type() {
	case cowrie.TypeFloat64:
		return v.Float64()
	case cowrie.TypeInt64:
		return float64(v.Int64())
	case cowrie.TypeUint64:
		return float64(v.Uint64())
	default:
		return 0
	}
}

func safeBool(v *cowrie.Value) bool {
	if v == nil {
		return false
	}
	switch v.Type() {
	case cowrie.TypeBool:
		return v.Bool()
	case cowrie.TypeInt64:
		return v.Int64() != 0
	case cowrie.TypeUint64:
		return v.Uint64() != 0
	default:
		return false
	}
}

// CowrieFormat implements Format using Cowrie binary encoding.
// Optimizes embedding storage using native tensor type (~48% smaller for float32 arrays).
type CowrieFormat struct {
	// UseCompression enables Zstd compression (additional ~40% reduction)
	UseCompression bool
}

func init() {
	Register(&CowrieFormat{UseCompression: false})
	Register(&CowrieFormat{UseCompression: true})
}

func (s *CowrieFormat) Name() string {
	if s.UseCompression {
		return "cowrie-zstd"
	}
	return "cowrie"
}

func (s *CowrieFormat) Extension() string {
	if s.UseCompression {
		return ".cowrie.zst"
	}
	return ".cowrie"
}

func (s *CowrieFormat) Save(w io.Writer, p *Payload) error {
	// Build Cowrie value with optimized tensor for embeddings
	obj := cowrie.Object()

	// Store embeddings as native tensor (main optimization)
	if len(p.Data) > 0 {
		obj.Set("data", encodeFloat32Tensor(p.Data))
	}

	// Scalar fields
	obj.Set("dim", cowrie.Int64(int64(p.Dim)))
	obj.Set("next", cowrie.Int64(p.Next))
	obj.Set("count", cowrie.Int64(int64(p.Count)))
	obj.Set("sum_doc_l", cowrie.Int64(int64(p.SumDocL)))
	obj.Set("checksum", cowrie.String(p.Checksum))
	obj.Set("last_saved", cowrie.String(p.LastSaved.Format(time.RFC3339Nano)))

	// String arrays
	obj.Set("docs", encodeStringArray(p.Docs))
	obj.Set("ids", encodeStringArray(p.IDs))

	// HNSW binary blob
	if len(p.HNSW) > 0 {
		obj.Set("hnsw", cowrie.Bytes(p.HNSW))
	}

	// Maps - encode as Cowrie objects
	obj.Set("meta", encodeStringMapMap(p.Meta))
	obj.Set("deleted", encodeBoolMap(p.Deleted))
	obj.Set("coll", encodeStringMapUint64(p.Coll))
	obj.Set("lex_tf", encodeIntMapMap(p.LexTF))
	obj.Set("doc_len", encodeIntMapUint64(p.DocLen))
	obj.Set("df", encodeStringIntMap(p.DF))
	obj.Set("num_meta", encodeFloat64MapMap(p.NumMeta))
	obj.Set("time_meta", encodeTimeMapMap(p.TimeMeta))

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

	// Decode embeddings tensor
	if dataVal := obj.Get("data"); dataVal != nil {
		p.Data = decodeFloat32Tensor(dataVal)
	}

	// Scalar fields - use safe accessors to prevent panics
	if v := obj.Get("dim"); v != nil {
		p.Dim = int(safeInt64(v))
	}
	if v := obj.Get("next"); v != nil {
		p.Next = safeInt64(v)
	}
	if v := obj.Get("count"); v != nil {
		p.Count = int(safeInt64(v))
	}
	if v := obj.Get("sum_doc_l"); v != nil {
		p.SumDocL = int(safeInt64(v))
	}
	if v := obj.Get("checksum"); v != nil {
		p.Checksum = safeString(v)
	}
	if v := obj.Get("last_saved"); v != nil {
		if t, err := time.Parse(time.RFC3339Nano, safeString(v)); err == nil {
			p.LastSaved = t
		}
	}

	// String arrays
	if v := obj.Get("docs"); v != nil {
		p.Docs = decodeStringArray(v)
	}
	if v := obj.Get("ids"); v != nil {
		p.IDs = decodeStringArray(v)
	}

	// HNSW binary
	if v := obj.Get("hnsw"); v != nil && v.Type() == cowrie.TypeBytes {
		p.HNSW = v.Bytes()
	}

	// Maps
	if v := obj.Get("meta"); v != nil {
		p.Meta = decodeStringMapMap(v)
	}
	if v := obj.Get("deleted"); v != nil {
		p.Deleted = decodeBoolMap(v)
	}
	if v := obj.Get("coll"); v != nil {
		p.Coll = decodeStringMapUint64(v)
	}
	if v := obj.Get("lex_tf"); v != nil {
		p.LexTF = decodeIntMapMap(v)
	}
	if v := obj.Get("doc_len"); v != nil {
		p.DocLen = decodeIntMapUint64(v)
	}
	if v := obj.Get("df"); v != nil {
		p.DF = decodeStringIntMap(v)
	}
	if v := obj.Get("num_meta"); v != nil {
		p.NumMeta = decodeFloat64MapMap(v)
	}
	if v := obj.Get("time_meta"); v != nil {
		p.TimeMeta = decodeTimeMapMap(v)
	}

	return p, nil
}

// Tensor encoding/decoding helpers

func encodeFloat32Tensor(values []float32) *cowrie.Value {
	data := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return cowrie.Tensor(cowrie.DTypeFloat32, []uint64{uint64(len(values))}, data)
}

func decodeFloat32Tensor(v *cowrie.Value) []float32 {
	if v.Type() != cowrie.TypeTensor {
		// Fallback for array type
		if v.Type() == cowrie.TypeArray {
			arr := v.Array()
			result := make([]float32, len(arr))
			for i, elem := range arr {
				result[i] = float32(safeFloat64(elem))
			}
			return result
		}
		return nil
	}

	td := v.Tensor()
	if td.DType != cowrie.DTypeFloat32 {
		return nil
	}

	count := len(td.Data) / 4
	result := make([]float32, count)
	for i := 0; i < count; i++ {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(td.Data[i*4:]))
	}
	return result
}

// String array helpers

func encodeStringArray(arr []string) *cowrie.Value {
	vals := make([]*cowrie.Value, len(arr))
	for i, s := range arr {
		vals[i] = cowrie.String(s)
	}
	return cowrie.Array(vals...)
}

func decodeStringArray(v *cowrie.Value) []string {
	if v.Type() != cowrie.TypeArray {
		return nil
	}
	arr := v.Array()
	result := make([]string, len(arr))
	for i, elem := range arr {
		result[i] = safeString(elem)
	}
	return result
}

// Map encoding/decoding helpers

func encodeStringMapMap(m map[uint64]map[string]string) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.String(iv))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeStringMapMap(v *cowrie.Value) map[uint64]map[string]string {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]string)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]string)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = safeString(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func encodeBoolMap(m map[uint64]bool) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), cowrie.Bool(v))
	}
	return obj
}

func decodeBoolMap(v *cowrie.Value) map[uint64]bool {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]bool)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = safeBool(m.Value)
	}
	return result
}

func encodeStringMapUint64(m map[uint64]string) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), cowrie.String(v))
	}
	return obj
}

func decodeStringMapUint64(v *cowrie.Value) map[uint64]string {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]string)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = safeString(m.Value)
	}
	return result
}

func encodeIntMapMap(m map[uint64]map[string]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.Int64(int64(iv)))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeIntMapMap(v *cowrie.Value) map[uint64]map[string]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]int)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]int)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = int(safeInt64(im.Value))
			}
		}
		result[k] = inner
	}
	return result
}

func encodeIntMapUint64(m map[uint64]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), cowrie.Int64(int64(v)))
	}
	return obj
}

func decodeIntMapUint64(v *cowrie.Value) map[uint64]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]int)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = int(safeInt64(m.Value))
	}
	return result
}

func encodeStringIntMap(m map[string]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(k, cowrie.Int64(int64(v)))
	}
	return obj
}

func decodeStringIntMap(v *cowrie.Value) map[string]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[string]int)
	for _, m := range v.Members() {
		result[m.Key] = int(safeInt64(m.Value))
	}
	return result
}

func encodeFloat64MapMap(m map[uint64]map[string]float64) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.Float64(iv))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeFloat64MapMap(v *cowrie.Value) map[uint64]map[string]float64 {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]float64)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]float64)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = safeFloat64(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func encodeTimeMapMap(m map[uint64]map[string]time.Time) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.String(iv.Format(time.RFC3339Nano)))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeTimeMapMap(v *cowrie.Value) map[uint64]map[string]time.Time {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]time.Time)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]time.Time)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				if t, err := time.Parse(time.RFC3339Nano, safeString(im.Value)); err == nil {
					inner[im.Key] = t
				}
			}
		}
		result[k] = inner
	}
	return result
}

// Key conversion helpers

func uint64Key(n uint64) string {
	return string(binary.BigEndian.AppendUint64(nil, n))
}

func parseUint64Key(s string) uint64 {
	if len(s) != 8 {
		return 0
	}
	return binary.BigEndian.Uint64([]byte(s))
}
