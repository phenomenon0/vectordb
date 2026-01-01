package index

import (
	"fmt"
	"strings"
)

// DefaultFactory is the standard index factory implementation.
// It maintains a registry of index constructors and supports dynamic registration.
type DefaultFactory struct {
	constructors map[string]IndexConstructor
}

// IndexConstructor is a function that creates a new index instance
type IndexConstructor func(dim int, config map[string]interface{}) (Index, error)

// NewFactory creates a new index factory with default implementations
func NewFactory() *DefaultFactory {
	f := &DefaultFactory{
		constructors: make(map[string]IndexConstructor),
	}
	// Register default index types here as they are implemented
	// f.Register("hnsw", NewHNSWIndex)
	// f.Register("ivf", NewIVFIndex)
	// f.Register("flat", NewFlatIndex)
	return f
}

// Register adds a new index constructor to the factory.
// This allows external packages to register custom index types.
//
// Parameters:
//   - indexType: Name of the index type (e.g., "hnsw", "ivf", "flat")
//   - constructor: Function that creates the index
//
// Returns:
//   - error if indexType is already registered
func (f *DefaultFactory) Register(indexType string, constructor IndexConstructor) error {
	indexType = strings.ToLower(indexType)
	if _, exists := f.constructors[indexType]; exists {
		return fmt.Errorf("index type %q already registered", indexType)
	}
	f.constructors[indexType] = constructor
	return nil
}

// Create creates a new index of the specified type.
//
// Parameters:
//   - indexType: Type of index to create (e.g., "hnsw", "ivf", "flat")
//   - dim: Vector dimensions (must be > 0)
//   - config: Index-specific configuration parameters
//
// Returns:
//   - Index: New index instance
//   - error: If indexType is unknown or creation fails
//
// Common config parameters (index-specific):
//   - HNSW: m (connections per node), ef_construction (build quality)
//   - IVF: nlist (number of clusters), nprobe (search clusters)
//   - FLAT: (no parameters)
func (f *DefaultFactory) Create(indexType string, dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	indexType = strings.ToLower(indexType)
	constructor, exists := f.constructors[indexType]
	if !exists {
		return nil, fmt.Errorf("unknown index type %q (supported: %v)",
			indexType, f.SupportedTypes())
	}

	return constructor(dim, config)
}

// SupportedTypes returns the list of registered index types
func (f *DefaultFactory) SupportedTypes() []string {
	types := make([]string, 0, len(f.constructors))
	for t := range f.constructors {
		types = append(types, t)
	}
	return types
}

// Global default factory instance (can be replaced for testing)
var globalFactory = NewFactory()

// Register is a convenience function that registers on the global factory
func Register(indexType string, constructor IndexConstructor) error {
	return globalFactory.Register(indexType, constructor)
}

// Create is a convenience function that creates using the global factory
func Create(indexType string, dim int, config map[string]interface{}) (Index, error) {
	return globalFactory.Create(indexType, dim, config)
}

// SupportedTypes is a convenience function that queries the global factory
func SupportedTypes() []string {
	return globalFactory.SupportedTypes()
}

// SetGlobalFactory replaces the global factory (useful for testing)
func SetGlobalFactory(f *DefaultFactory) {
	globalFactory = f
}

// GetConfigInt extracts an integer from config map with default value
func GetConfigInt(config map[string]interface{}, key string, defaultValue int) int {
	if config == nil {
		return defaultValue
	}
	if v, ok := config[key]; ok {
		switch val := v.(type) {
		case int:
			return val
		case int64:
			return int(val)
		case float64:
			return int(val)
		}
	}
	return defaultValue
}

// GetConfigInt64 extracts an int64 from config map with default value
func GetConfigInt64(config map[string]interface{}, key string, defaultValue int64) int64 {
	if config == nil {
		return defaultValue
	}
	if v, ok := config[key]; ok {
		switch val := v.(type) {
		case int64:
			return val
		case int:
			return int64(val)
		case float64:
			return int64(val)
		}
	}
	return defaultValue
}

// GetConfigFloat extracts a float from config map with default value
func GetConfigFloat(config map[string]interface{}, key string, defaultValue float64) float64 {
	if config == nil {
		return defaultValue
	}
	if v, ok := config[key]; ok {
		switch val := v.(type) {
		case float64:
			return val
		case float32:
			return float64(val)
		case int:
			return float64(val)
		}
	}
	return defaultValue
}

// GetConfigString extracts a string from config map with default value
func GetConfigString(config map[string]interface{}, key string, defaultValue string) string {
	if config == nil {
		return defaultValue
	}
	if v, ok := config[key]; ok {
		if str, ok := v.(string); ok {
			return str
		}
	}
	return defaultValue
}

// GetConfigBool extracts a boolean from config map with default value
func GetConfigBool(config map[string]interface{}, key string, defaultValue bool) bool {
	if config == nil {
		return defaultValue
	}
	if v, ok := config[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return defaultValue
}
