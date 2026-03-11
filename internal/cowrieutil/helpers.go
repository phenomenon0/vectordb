// Package cowrieutil provides shared cowrie encoding/decoding helpers
// used by the WAL, snapshot, and storage packages.
package cowrieutil

import (
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/Neumenon/cowrie/go"
)

// Safe value extraction helpers — prevent panics from type mismatches

func SafeString(v *cowrie.Value) string {
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

func SafeInt64(v *cowrie.Value) int64 {
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

func SafeUint64(v *cowrie.Value) uint64 {
	if v == nil {
		return 0
	}
	switch v.Type() {
	case cowrie.TypeUint64:
		return v.Uint64()
	case cowrie.TypeInt64:
		return uint64(v.Int64())
	case cowrie.TypeFloat64:
		return uint64(v.Float64())
	default:
		return 0
	}
}

func SafeFloat64(v *cowrie.Value) float64 {
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

func SafeBool(v *cowrie.Value) bool {
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

// Tensor encoding/decoding

func EncodeFloat32Tensor(values []float32) *cowrie.Value {
	data := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return cowrie.Tensor(cowrie.DTypeFloat32, []uint64{uint64(len(values))}, data)
}

func DecodeFloat32Tensor(v *cowrie.Value) []float32 {
	if v.Type() != cowrie.TypeTensor {
		if v.Type() == cowrie.TypeArray {
			arr := v.Array()
			result := make([]float32, len(arr))
			for i, elem := range arr {
				result[i] = float32(SafeFloat64(elem))
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

func EncodeStringArray(arr []string) *cowrie.Value {
	vals := make([]*cowrie.Value, len(arr))
	for i, s := range arr {
		vals[i] = cowrie.String(s)
	}
	return cowrie.Array(vals...)
}

func DecodeStringArray(v *cowrie.Value) []string {
	if v.Type() != cowrie.TypeArray {
		return nil
	}
	arr := v.Array()
	result := make([]string, len(arr))
	for i, elem := range arr {
		result[i] = SafeString(elem)
	}
	return result
}

// Uint64 array helpers

func EncodeUint64Array(arr []uint64) *cowrie.Value {
	vals := make([]*cowrie.Value, len(arr))
	for i, n := range arr {
		vals[i] = cowrie.Int64(int64(n))
	}
	return cowrie.Array(vals...)
}

func DecodeUint64Array(v *cowrie.Value) []uint64 {
	if v.Type() != cowrie.TypeArray {
		return nil
	}
	arr := v.Array()
	result := make([]uint64, len(arr))
	for i, elem := range arr {
		result[i] = SafeUint64(elem)
	}
	return result
}

// Key conversion — uint64 keys encoded as big-endian 8-byte strings

func Uint64Key(n uint64) string {
	return string(binary.BigEndian.AppendUint64(nil, n))
}

func ParseUint64Key(s string) uint64 {
	if len(s) != 8 {
		return 0
	}
	return binary.BigEndian.Uint64([]byte(s))
}

// Map encoding/decoding helpers

func EncodeStringMapMap(m map[uint64]map[string]string) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.String(iv))
		}
		obj.Set(Uint64Key(k), inner)
	}
	return obj
}

func DecodeStringMapMap(v *cowrie.Value) map[uint64]map[string]string {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]string)
	for _, m := range v.Members() {
		k := ParseUint64Key(m.Key)
		inner := make(map[string]string)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = SafeString(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func EncodeBoolMap(m map[uint64]bool) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(Uint64Key(k), cowrie.Bool(v))
	}
	return obj
}

func DecodeBoolMap(v *cowrie.Value) map[uint64]bool {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]bool)
	for _, m := range v.Members() {
		result[ParseUint64Key(m.Key)] = SafeBool(m.Value)
	}
	return result
}

func EncodeStringMapUint64(m map[uint64]string) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(Uint64Key(k), cowrie.String(v))
	}
	return obj
}

func DecodeStringMapUint64(v *cowrie.Value) map[uint64]string {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]string)
	for _, m := range v.Members() {
		result[ParseUint64Key(m.Key)] = SafeString(m.Value)
	}
	return result
}

func EncodeIntMapMap(m map[uint64]map[string]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.Int64(int64(iv)))
		}
		obj.Set(Uint64Key(k), inner)
	}
	return obj
}

func DecodeIntMapMap(v *cowrie.Value) map[uint64]map[string]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]int)
	for _, m := range v.Members() {
		k := ParseUint64Key(m.Key)
		inner := make(map[string]int)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = int(SafeInt64(im.Value))
			}
		}
		result[k] = inner
	}
	return result
}

func EncodeIntMapUint64(m map[uint64]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(Uint64Key(k), cowrie.Int64(int64(v)))
	}
	return obj
}

func DecodeIntMapUint64(v *cowrie.Value) map[uint64]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]int)
	for _, m := range v.Members() {
		result[ParseUint64Key(m.Key)] = int(SafeInt64(m.Value))
	}
	return result
}

func EncodeStringIntMap(m map[string]int) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(k, cowrie.Int64(int64(v)))
	}
	return obj
}

func DecodeStringIntMap(v *cowrie.Value) map[string]int {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[string]int)
	for _, m := range v.Members() {
		result[m.Key] = int(SafeInt64(m.Value))
	}
	return result
}

func EncodeFloat64MapMap(m map[uint64]map[string]float64) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.Float64(iv))
		}
		obj.Set(Uint64Key(k), inner)
	}
	return obj
}

func DecodeFloat64MapMap(v *cowrie.Value) map[uint64]map[string]float64 {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]float64)
	for _, m := range v.Members() {
		k := ParseUint64Key(m.Key)
		inner := make(map[string]float64)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				inner[im.Key] = SafeFloat64(im.Value)
			}
		}
		result[k] = inner
	}
	return result
}

func EncodeTimeMapMap(m map[uint64]map[string]time.Time) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		inner := cowrie.Object()
		for ik, iv := range v {
			inner.Set(ik, cowrie.String(iv.Format(time.RFC3339Nano)))
		}
		obj.Set(Uint64Key(k), inner)
	}
	return obj
}

func DecodeTimeMapMap(v *cowrie.Value) map[uint64]map[string]time.Time {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[uint64]map[string]time.Time)
	for _, m := range v.Members() {
		k := ParseUint64Key(m.Key)
		inner := make(map[string]time.Time)
		if m.Value.Type() == cowrie.TypeObject {
			for _, im := range m.Value.Members() {
				if t, err := time.Parse(time.RFC3339Nano, SafeString(im.Value)); err == nil {
					inner[im.Key] = t
				}
			}
		}
		result[k] = inner
	}
	return result
}

// EncodeStringStringMap encodes map[string]string for WAL metadata.
func EncodeStringStringMap(m map[string]string) *cowrie.Value {
	if m == nil {
		return cowrie.Object()
	}
	obj := cowrie.Object()
	for k, v := range m {
		obj.Set(k, cowrie.String(v))
	}
	return obj
}

func DecodeStringStringMap(v *cowrie.Value) map[string]string {
	if v.Type() != cowrie.TypeObject {
		return nil
	}
	result := make(map[string]string)
	for _, m := range v.Members() {
		result[m.Key] = SafeString(m.Value)
	}
	return result
}
