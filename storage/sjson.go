package storage

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"time"

	"github.com/phenomenon0/Agent-GO/sjson"
)

// Safe value extraction helpers - prevent panics from type mismatches

func safeString(v *sjson.Value) string {
	if v == nil {
		return ""
	}
	switch v.Type() {
	case sjson.TypeString:
		return v.String()
	case sjson.TypeInt64:
		return fmt.Sprintf("%d", v.Int64())
	case sjson.TypeUint64:
		return fmt.Sprintf("%d", v.Uint64())
	case sjson.TypeFloat64:
		return fmt.Sprintf("%g", v.Float64())
	case sjson.TypeBool:
		return fmt.Sprintf("%t", v.Bool())
	default:
		return fmt.Sprint(sjson.ToAny(v))
	}
}

func safeInt64(v *sjson.Value) int64 {
	if v == nil {
		return 0
	}
	switch v.Type() {
	case sjson.TypeInt64:
		return v.Int64()
	case sjson.TypeUint64:
		return int64(v.Uint64())
	case sjson.TypeFloat64:
		return int64(v.Float64())
	default:
		return 0
	}
}

func safeFloat64(v *sjson.Value) float64 {
	if v == nil {
		return 0
	}
	switch v.Type() {
	case sjson.TypeFloat64:
		return v.Float64()
	case sjson.TypeInt64:
		return float64(v.Int64())
	case sjson.TypeUint64:
		return float64(v.Uint64())
	default:
		return 0
	}
}

func safeBool(v *sjson.Value) bool {
	if v == nil {
		return false
	}
	switch v.Type() {
	case sjson.TypeBool:
		return v.Bool()
	case sjson.TypeInt64:
		return v.Int64() != 0
	case sjson.TypeUint64:
		return v.Uint64() != 0
	default:
		return false
	}
}

// SJSONFormat implements Format using SJSON binary encoding.
// Optimizes embedding storage using native tensor type (~48% smaller for float32 arrays).
type SJSONFormat struct {
	// UseCompression enables Zstd compression (additional ~40% reduction)
	UseCompression bool
}

func init() {
	Register(&SJSONFormat{UseCompression: false})
	Register(&SJSONFormat{UseCompression: true})
}

func (s *SJSONFormat) Name() string {
	if s.UseCompression {
		return "sjson-zstd"
	}
	return "sjson"
}

func (s *SJSONFormat) Extension() string {
	if s.UseCompression {
		return ".sjson.zst"
	}
	return ".sjson"
}

func (s *SJSONFormat) Save(w io.Writer, p *Payload) error {
	// Build SJSON value with optimized tensor for embeddings
	obj := sjson.Object()

	// Store embeddings as native tensor (main optimization)
	if len(p.Data) > 0 {
		obj.Set("data", encodeFloat32Tensor(p.Data))
	}

	// Scalar fields
	obj.Set("dim", sjson.Int64(int64(p.Dim)))
	obj.Set("next", sjson.Int64(p.Next))
	obj.Set("count", sjson.Int64(int64(p.Count)))
	obj.Set("sum_doc_l", sjson.Int64(int64(p.SumDocL)))
	obj.Set("checksum", sjson.String(p.Checksum))
	obj.Set("last_saved", sjson.String(p.LastSaved.Format(time.RFC3339Nano)))

	// String arrays
	obj.Set("docs", encodeStringArray(p.Docs))
	obj.Set("ids", encodeStringArray(p.IDs))

	// HNSW binary blob
	if len(p.HNSW) > 0 {
		obj.Set("hnsw", sjson.Bytes(p.HNSW))
	}

	// Maps - encode as SJSON objects
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
		data, err = sjson.EncodeFramed(obj, sjson.CompressionZstd)
	} else {
		data, err = sjson.Encode(obj)
	}

	if err != nil {
		return err
	}

	_, err = w.Write(data)
	return err
}

func (s *SJSONFormat) Load(r io.Reader) (*Payload, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	var obj *sjson.Value
	if s.UseCompression {
		obj, err = sjson.DecodeFramed(data)
	} else {
		obj, err = sjson.Decode(data)
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
	if v := obj.Get("hnsw"); v != nil && v.Type() == sjson.TypeBytes {
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

func encodeFloat32Tensor(values []float32) *sjson.Value {
	data := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return sjson.Tensor(sjson.DTypeFloat32, []uint64{uint64(len(values))}, data)
}

func decodeFloat32Tensor(v *sjson.Value) []float32 {
	if v.Type() != sjson.TypeTensor {
		// Fallback for array type
		if v.Type() == sjson.TypeArray {
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
	if td.DType != sjson.DTypeFloat32 {
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

func encodeStringArray(arr []string) *sjson.Value {
	vals := make([]*sjson.Value, len(arr))
	for i, s := range arr {
		vals[i] = sjson.String(s)
	}
	return sjson.Array(vals...)
}

func decodeStringArray(v *sjson.Value) []string {
	if v.Type() != sjson.TypeArray {
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

func encodeStringMapMap(m map[uint64]map[string]string) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		inner := sjson.Object()
		for ik, iv := range v {
			inner.Set(ik, sjson.String(iv))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeStringMapMap(v *sjson.Value) map[uint64]map[string]string {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]string)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]string)
		if m.Value.Type() == sjson.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = safeString(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func encodeBoolMap(m map[uint64]bool) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), sjson.Bool(v))
	}
	return obj
}

func decodeBoolMap(v *sjson.Value) map[uint64]bool {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]bool)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = safeBool(m.Value)
	}
	return result
}

func encodeStringMapUint64(m map[uint64]string) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), sjson.String(v))
	}
	return obj
}

func decodeStringMapUint64(v *sjson.Value) map[uint64]string {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]string)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = safeString(m.Value)
	}
	return result
}

func encodeIntMapMap(m map[uint64]map[string]int) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		inner := sjson.Object()
		for ik, iv := range v {
			inner.Set(ik, sjson.Int64(int64(iv)))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeIntMapMap(v *sjson.Value) map[uint64]map[string]int {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]int)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]int)
		if m.Value.Type() == sjson.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = int(safeInt64(im.Value))
			}
		}
		result[k] = inner
	}
	return result
}

func encodeIntMapUint64(m map[uint64]int) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		obj.Set(uint64Key(k), sjson.Int64(int64(v)))
	}
	return obj
}

func decodeIntMapUint64(v *sjson.Value) map[uint64]int {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]int)
	for _, m := range v.Members() {
		result[parseUint64Key(m.Key)] = int(safeInt64(m.Value))
	}
	return result
}

func encodeStringIntMap(m map[string]int) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		obj.Set(k, sjson.Int64(int64(v)))
	}
	return obj
}

func decodeStringIntMap(v *sjson.Value) map[string]int {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[string]int)
	for _, m := range v.Members() {
		result[m.Key] = int(safeInt64(m.Value))
	}
	return result
}

func encodeFloat64MapMap(m map[uint64]map[string]float64) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		inner := sjson.Object()
		for ik, iv := range v {
			inner.Set(ik, sjson.Float64(iv))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeFloat64MapMap(v *sjson.Value) map[uint64]map[string]float64 {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]float64)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]float64)
		if m.Value.Type() == sjson.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = safeFloat64(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func encodeTimeMapMap(m map[uint64]map[string]time.Time) *sjson.Value {
	if m == nil {
		return sjson.Object()
	}
	obj := sjson.Object()
	for k, v := range m {
		inner := sjson.Object()
		for ik, iv := range v {
			inner.Set(ik, sjson.String(iv.Format(time.RFC3339Nano)))
		}
		obj.Set(uint64Key(k), inner)
	}
	return obj
}

func decodeTimeMapMap(v *sjson.Value) map[uint64]map[string]time.Time {
	if v.Type() != sjson.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]time.Time)
	for _, m := range v.Members() {
		k := parseUint64Key(m.Key)
		inner := make(map[string]time.Time)
		if m.Value.Type() == sjson.TypeObject {
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
