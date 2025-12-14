package filter

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Filter represents a metadata filter for vector search
type Filter interface {
	// Evaluate checks if the given metadata matches the filter
	Evaluate(metadata map[string]interface{}) bool
	// String returns a human-readable representation
	String() string
}

// Operator represents filter operator types
type Operator string

const (
	OpEqual              Operator = "eq"
	OpNotEqual           Operator = "ne"
	OpGreaterThan        Operator = "gt"
	OpGreaterThanOrEqual Operator = "gte"
	OpLessThan           Operator = "lt"
	OpLessThanOrEqual    Operator = "lte"
	OpIn                 Operator = "in"
	OpNotIn              Operator = "nin"
	OpContains           Operator = "contains"
	OpStartsWith         Operator = "startswith"
	OpEndsWith           Operator = "endswith"
	OpRegex              Operator = "regex"
	OpExists             Operator = "exists"
)

// ComparisonFilter compares a field to a value
type ComparisonFilter struct {
	Field    string      // JSON path to field (e.g., "product.specs.color")
	Operator Operator    // Comparison operator
	Value    interface{} // Value to compare against
}

// Evaluate checks if metadata matches the comparison
func (f *ComparisonFilter) Evaluate(metadata map[string]interface{}) bool {
	fieldValue, exists := getNestedValue(metadata, f.Field)

	switch f.Operator {
	case OpExists:
		boolVal, ok := f.Value.(bool)
		if !ok {
			return false
		}
		return exists == boolVal

	case OpEqual:
		if !exists {
			return false
		}
		return compareEqual(fieldValue, f.Value)

	case OpNotEqual:
		if !exists {
			return true // Non-existent field is not equal
		}
		return !compareEqual(fieldValue, f.Value)

	case OpGreaterThan:
		if !exists {
			return false
		}
		return compareGreaterThan(fieldValue, f.Value)

	case OpGreaterThanOrEqual:
		if !exists {
			return false
		}
		return compareGreaterThan(fieldValue, f.Value) || compareEqual(fieldValue, f.Value)

	case OpLessThan:
		if !exists {
			return false
		}
		return compareLessThan(fieldValue, f.Value)

	case OpLessThanOrEqual:
		if !exists {
			return false
		}
		return compareLessThan(fieldValue, f.Value) || compareEqual(fieldValue, f.Value)

	case OpIn:
		if !exists {
			return false
		}
		return compareIn(fieldValue, f.Value)

	case OpNotIn:
		if !exists {
			return true
		}
		return !compareIn(fieldValue, f.Value)

	case OpContains:
		if !exists {
			return false
		}
		return compareContains(fieldValue, f.Value)

	case OpStartsWith:
		if !exists {
			return false
		}
		return compareStartsWith(fieldValue, f.Value)

	case OpEndsWith:
		if !exists {
			return false
		}
		return compareEndsWith(fieldValue, f.Value)

	case OpRegex:
		if !exists {
			return false
		}
		return compareRegex(fieldValue, f.Value)

	default:
		return false
	}
}

func (f *ComparisonFilter) String() string {
	return fmt.Sprintf("%s %s %v", f.Field, f.Operator, f.Value)
}

// AndFilter combines filters with AND logic (all must match)
type AndFilter struct {
	Filters []Filter
}

func (f *AndFilter) Evaluate(metadata map[string]interface{}) bool {
	for _, filter := range f.Filters {
		if !filter.Evaluate(metadata) {
			return false
		}
	}
	return true
}

func (f *AndFilter) String() string {
	parts := make([]string, len(f.Filters))
	for i, filter := range f.Filters {
		parts[i] = filter.String()
	}
	return fmt.Sprintf("AND(%s)", strings.Join(parts, ", "))
}

// OrFilter combines filters with OR logic (at least one must match)
type OrFilter struct {
	Filters []Filter
}

func (f *OrFilter) Evaluate(metadata map[string]interface{}) bool {
	for _, filter := range f.Filters {
		if filter.Evaluate(metadata) {
			return true
		}
	}
	return false
}

func (f *OrFilter) String() string {
	parts := make([]string, len(f.Filters))
	for i, filter := range f.Filters {
		parts[i] = filter.String()
	}
	return fmt.Sprintf("OR(%s)", strings.Join(parts, ", "))
}

// NotFilter inverts a filter
type NotFilter struct {
	Filter Filter
}

func (f *NotFilter) Evaluate(metadata map[string]interface{}) bool {
	return !f.Filter.Evaluate(metadata)
}

func (f *NotFilter) String() string {
	return fmt.Sprintf("NOT(%s)", f.Filter.String())
}

// Helper functions

// getNestedValue retrieves a value from nested JSON using dot notation
// Example: "product.specs.color" -> metadata["product"]["specs"]["color"]
func getNestedValue(data map[string]interface{}, path string) (interface{}, bool) {
	parts := strings.Split(path, ".")
	current := data

	for i, part := range parts {
		value, exists := current[part]
		if !exists {
			return nil, false
		}

		// Last part - return the value
		if i == len(parts)-1 {
			return value, true
		}

		// Not last part - must be a map
		nestedMap, ok := value.(map[string]interface{})
		if !ok {
			return nil, false
		}
		current = nestedMap
	}

	return nil, false
}

// compareEqual checks if two values are equal
func compareEqual(a, b interface{}) bool {
	// Handle nil
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Try direct comparison
	if reflect.DeepEqual(a, b) {
		return true
	}

	// Try numeric comparison (handle float64/int conversions)
	aNum, aOk := toFloat64(a)
	bNum, bOk := toFloat64(b)
	if aOk && bOk {
		return aNum == bNum
	}

	// Try string comparison
	aStr := fmt.Sprintf("%v", a)
	bStr := fmt.Sprintf("%v", b)
	return aStr == bStr
}

// compareGreaterThan checks if a > b
func compareGreaterThan(a, b interface{}) bool {
	// Try numeric comparison
	aNum, aOk := toFloat64(a)
	bNum, bOk := toFloat64(b)
	if aOk && bOk {
		return aNum > bNum
	}

	// Try string comparison
	aStr, aOk := a.(string)
	bStr, bOk := b.(string)
	if aOk && bOk {
		return aStr > bStr
	}

	// Try time comparison
	aTime, aOk := parseTime(a)
	bTime, bOk := parseTime(b)
	if aOk && bOk {
		return aTime.After(bTime)
	}

	return false
}

// compareLessThan checks if a < b
func compareLessThan(a, b interface{}) bool {
	// Try numeric comparison
	aNum, aOk := toFloat64(a)
	bNum, bOk := toFloat64(b)
	if aOk && bOk {
		return aNum < bNum
	}

	// Try string comparison
	aStr, aOk := a.(string)
	bStr, bOk := b.(string)
	if aOk && bOk {
		return aStr < bStr
	}

	// Try time comparison
	aTime, aOk := parseTime(a)
	bTime, bOk := parseTime(b)
	if aOk && bOk {
		return aTime.Before(bTime)
	}

	return false
}

// compareIn checks if value is in array
func compareIn(value, array interface{}) bool {
	// Convert array to slice
	arraySlice, ok := toSlice(array)
	if !ok {
		return false
	}

	// Check if value is in slice
	for _, item := range arraySlice {
		if compareEqual(value, item) {
			return true
		}
	}

	return false
}

// compareContains checks if value contains substring or array contains element
func compareContains(haystack, needle interface{}) bool {
	// String contains
	haystackStr, ok := haystack.(string)
	needleStr, nok := needle.(string)
	if ok && nok {
		return strings.Contains(haystackStr, needleStr)
	}

	// Array contains
	haystackSlice, ok := toSlice(haystack)
	if ok {
		for _, item := range haystackSlice {
			if compareEqual(item, needle) {
				return true
			}
		}
	}

	return false
}

// compareStartsWith checks if string starts with prefix
func compareStartsWith(value, prefix interface{}) bool {
	valueStr, ok := value.(string)
	prefixStr, pok := prefix.(string)
	if !ok || !pok {
		return false
	}
	return strings.HasPrefix(valueStr, prefixStr)
}

// compareEndsWith checks if string ends with suffix
func compareEndsWith(value, suffix interface{}) bool {
	valueStr, ok := value.(string)
	suffixStr, sok := suffix.(string)
	if !ok || !sok {
		return false
	}
	return strings.HasSuffix(valueStr, suffixStr)
}

// compareRegex checks if value matches regex pattern
func compareRegex(value, pattern interface{}) bool {
	valueStr, ok := value.(string)
	patternStr, pok := pattern.(string)
	if !ok || !pok {
		return false
	}

	matched, err := regexp.MatchString(patternStr, valueStr)
	if err != nil {
		return false
	}
	return matched
}

// toFloat64 converts various numeric types to float64
func toFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int32:
		return float64(val), true
	case int64:
		return float64(val), true
	case uint:
		return float64(val), true
	case uint32:
		return float64(val), true
	case uint64:
		return float64(val), true
	case string:
		f, err := strconv.ParseFloat(val, 64)
		return f, err == nil
	default:
		return 0, false
	}
}

// toSlice converts various array types to []interface{}
func toSlice(v interface{}) ([]interface{}, bool) {
	switch val := v.(type) {
	case []interface{}:
		return val, true
	case []string:
		result := make([]interface{}, len(val))
		for i, s := range val {
			result[i] = s
		}
		return result, true
	case []int:
		result := make([]interface{}, len(val))
		for i, n := range val {
			result[i] = n
		}
		return result, true
	case []float64:
		result := make([]interface{}, len(val))
		for i, n := range val {
			result[i] = n
		}
		return result, true
	default:
		// Try reflection
		rv := reflect.ValueOf(v)
		if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
			result := make([]interface{}, rv.Len())
			for i := 0; i < rv.Len(); i++ {
				result[i] = rv.Index(i).Interface()
			}
			return result, true
		}
		return nil, false
	}
}

// parseTime attempts to parse various time formats
func parseTime(v interface{}) (time.Time, bool) {
	switch val := v.(type) {
	case time.Time:
		return val, true
	case string:
		// Try common formats
		formats := []string{
			time.RFC3339,
			"2006-01-02",
			"2006-01-02 15:04:05",
			time.RFC1123,
		}
		for _, format := range formats {
			t, err := time.Parse(format, val)
			if err == nil {
				return t, true
			}
		}
		return time.Time{}, false
	default:
		return time.Time{}, false
	}
}

// Builder functions for convenient filter construction

// Eq creates an equality filter
func Eq(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpEqual, Value: value}
}

// Ne creates a not-equal filter
func Ne(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpNotEqual, Value: value}
}

// Gt creates a greater-than filter
func Gt(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpGreaterThan, Value: value}
}

// Gte creates a greater-than-or-equal filter
func Gte(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpGreaterThanOrEqual, Value: value}
}

// Lt creates a less-than filter
func Lt(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpLessThan, Value: value}
}

// Lte creates a less-than-or-equal filter
func Lte(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpLessThanOrEqual, Value: value}
}

// In creates an in-array filter
func In(field string, values interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpIn, Value: values}
}

// NotIn creates a not-in-array filter
func NotIn(field string, values interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpNotIn, Value: values}
}

// Contains creates a contains filter
func Contains(field string, value interface{}) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpContains, Value: value}
}

// StartsWith creates a starts-with filter
func StartsWith(field string, value string) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpStartsWith, Value: value}
}

// EndsWith creates an ends-with filter
func EndsWith(field string, value string) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpEndsWith, Value: value}
}

// Regex creates a regex filter
func Regex(field string, pattern string) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpRegex, Value: pattern}
}

// Exists creates an exists filter
func Exists(field string) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpExists, Value: true}
}

// NotExists creates a not-exists filter
func NotExists(field string) *ComparisonFilter {
	return &ComparisonFilter{Field: field, Operator: OpExists, Value: false}
}

// And combines filters with AND logic
func And(filters ...Filter) *AndFilter {
	return &AndFilter{Filters: filters}
}

// Or combines filters with OR logic
func Or(filters ...Filter) *OrFilter {
	return &OrFilter{Filters: filters}
}

// Not inverts a filter
func Not(filter Filter) *NotFilter {
	return &NotFilter{Filter: filter}
}

// FromJSON parses a filter from JSON
func FromJSON(data []byte) (Filter, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return parseFilterMap(raw)
}

// FromMap parses a filter from a map (e.g., from HTTP request)
func FromMap(data map[string]interface{}) (Filter, error) {
	if data == nil || len(data) == 0 {
		return nil, nil // No filter
	}
	return parseFilterMap(data)
}

// parseFilterMap recursively parses a filter from a map
func parseFilterMap(data map[string]interface{}) (Filter, error) {
	// Check for logical operators
	if and, ok := data["$and"]; ok {
		filters, err := parseFilterArray(and)
		if err != nil {
			return nil, err
		}
		return &AndFilter{Filters: filters}, nil
	}

	if or, ok := data["$or"]; ok {
		filters, err := parseFilterArray(or)
		if err != nil {
			return nil, err
		}
		return &OrFilter{Filters: filters}, nil
	}

	if not, ok := data["$not"]; ok {
		notMap, ok := not.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("$not must be an object")
		}
		filter, err := parseFilterMap(notMap)
		if err != nil {
			return nil, err
		}
		return &NotFilter{Filter: filter}, nil
	}

	// Parse as comparison filter
	// Format: {"field": {"$op": value}}
	for field, value := range data {
		opMap, ok := value.(map[string]interface{})
		if !ok {
			// Simple equality: {"field": value}
			return &ComparisonFilter{Field: field, Operator: OpEqual, Value: value}, nil
		}

		// Parse operator
		for opStr, val := range opMap {
			op := Operator(strings.TrimPrefix(opStr, "$"))
			return &ComparisonFilter{Field: field, Operator: op, Value: val}, nil
		}
	}

	return nil, fmt.Errorf("invalid filter format")
}

// parseFilterArray parses an array of filters
func parseFilterArray(data interface{}) ([]Filter, error) {
	arr, ok := data.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array")
	}

	filters := make([]Filter, len(arr))
	for i, item := range arr {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("array item must be object")
		}
		filter, err := parseFilterMap(itemMap)
		if err != nil {
			return nil, err
		}
		filters[i] = filter
	}

	return filters, nil
}
