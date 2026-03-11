package filter

import (
	"testing"
	"time"
)

// TestComparisonFilters tests basic comparison operations
func TestComparisonFilters(t *testing.T) {
	metadata := map[string]interface{}{
		"price":    100.0,
		"category": "electronics",
		"stock":    25,
		"featured": true,
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Equal string", Eq("category", "electronics"), true},
		{"Equal number", Eq("price", 100.0), true},
		{"Equal bool", Eq("featured", true), true},
		{"Not equal", Ne("category", "books"), true},
		{"Greater than", Gt("price", 50.0), true},
		{"Greater than false", Gt("price", 150.0), false},
		{"Greater than or equal", Gte("price", 100.0), true},
		{"Less than", Lt("stock", 30), true},
		{"Less than false", Lt("stock", 20), false},
		{"Less than or equal", Lte("stock", 25), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestArrayOperations tests IN and array operations
func TestArrayOperations(t *testing.T) {
	metadata := map[string]interface{}{
		"tags":      []string{"electronics", "phone", "mobile"},
		"colors":    []interface{}{"black", "white", "blue"},
		"ratings":   []int{4, 5, 3},
		"category":  "electronics",
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"In array string", In("category", []string{"electronics", "books"}), true},
		{"In array miss", In("category", []string{"books", "toys"}), false},
		{"Not in array", NotIn("category", []string{"books", "toys"}), true},
		{"Contains string slice", Contains("tags", "phone"), true},
		{"Contains string slice miss", Contains("tags", "laptop"), false},
		{"Contains interface slice", Contains("colors", "black"), true},
		{"Contains int slice", Contains("ratings", 5), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestStringOperations tests string-specific operations
func TestStringOperations(t *testing.T) {
	metadata := map[string]interface{}{
		"title":       "iPhone 15 Pro Max",
		"sku":         "PROD-12345",
		"description": "Latest flagship smartphone with advanced features",
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Starts with", StartsWith("title", "iPhone"), true},
		{"Starts with false", StartsWith("title", "Samsung"), false},
		{"Ends with", EndsWith("sku", "12345"), true},
		{"Ends with false", EndsWith("sku", "99999"), false},
		{"Contains substring", Contains("description", "smartphone"), true},
		{"Contains substring false", Contains("description", "laptop"), false},
		{"Regex match", Regex("sku", "^PROD-\\d+$"), true},
		{"Regex no match", Regex("sku", "^ORDER-"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestNestedJSONPaths tests nested field access
func TestNestedJSONPaths(t *testing.T) {
	metadata := map[string]interface{}{
		"product": map[string]interface{}{
			"name": "Laptop",
			"specs": map[string]interface{}{
				"cpu":    "Intel i7",
				"ram":    16,
				"color":  "silver",
			},
			"pricing": map[string]interface{}{
				"regular": 1200.0,
				"sale":    999.0,
			},
		},
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Nested level 1", Eq("product.name", "Laptop"), true},
		{"Nested level 2", Eq("product.specs.cpu", "Intel i7"), true},
		{"Nested level 2 number", Eq("product.specs.ram", 16), true},
		{"Nested level 2 color", Eq("product.specs.color", "silver"), true},
		{"Nested pricing", Gt("product.pricing.sale", 900.0), true},
		{"Nested pricing range", Lt("product.pricing.regular", 1500.0), true},
		{"Non-existent nested", Eq("product.specs.gpu", "NVIDIA"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestBooleanLogic tests AND/OR/NOT combinations
func TestBooleanLogic(t *testing.T) {
	metadata := map[string]interface{}{
		"price":    150.0,
		"category": "electronics",
		"stock":    10,
		"featured": true,
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{
			"AND all true",
			And(
				Eq("category", "electronics"),
				Gt("price", 100.0),
				Lt("stock", 20),
			),
			true,
		},
		{
			"AND one false",
			And(
				Eq("category", "electronics"),
				Gt("price", 200.0), // False
				Lt("stock", 20),
			),
			false,
		},
		{
			"OR all false",
			Or(
				Eq("category", "books"),
				Gt("price", 200.0),
			),
			false,
		},
		{
			"OR one true",
			Or(
				Eq("category", "books"),      // False
				Gt("price", 100.0),           // True
				Eq("featured", false),        // False
			),
			true,
		},
		{
			"NOT true becomes false",
			Not(Eq("category", "electronics")),
			false,
		},
		{
			"NOT false becomes true",
			Not(Eq("category", "books")),
			true,
		},
		{
			"Complex: (price > 100 AND category = electronics) OR stock < 5",
			Or(
				And(
					Gt("price", 100.0),
					Eq("category", "electronics"),
				),
				Lt("stock", 5),
			),
			true,
		},
		{
			"Complex: NOT ((price > 200) OR (stock < 5))",
			Not(
				Or(
					Gt("price", 200.0),
					Lt("stock", 5),
				),
			),
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestExistsFilter tests field existence checks
func TestExistsFilter(t *testing.T) {
	metadata := map[string]interface{}{
		"name":  "Product",
		"price": 100.0,
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Exists true", Exists("name"), true},
		{"Exists false", Exists("description"), false},
		{"Not exists", NotExists("description"), true},
		{"Not exists false", NotExists("name"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestDateTimeComparisons tests date/time operations
func TestDateTimeComparisons(t *testing.T) {
	now := time.Now()
	yesterday := now.Add(-24 * time.Hour)
	tomorrow := now.Add(24 * time.Hour)

	metadata := map[string]interface{}{
		"created_at": now.Format(time.RFC3339),
		"expires_at": tomorrow.Format("2006-01-02"),
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Date after yesterday", Gt("created_at", yesterday.Format(time.RFC3339)), true},
		{"Date before tomorrow", Lt("created_at", tomorrow.Format(time.RFC3339)), true},
		{"Date equals", Eq("expires_at", tomorrow.Format("2006-01-02")), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestNumericTypeConversions tests handling of different numeric types
func TestNumericTypeConversions(t *testing.T) {
	metadata := map[string]interface{}{
		"int_val":     42,
		"int64_val":   int64(100),
		"float32_val": float32(3.14),
		"float64_val": 2.718,
		"string_num":  "123.45",
	}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"Int equals float", Eq("int_val", 42.0), true},
		{"Int64 greater than", Gt("int64_val", 50), true},
		{"Float32 less than", Lt("float32_val", 4.0), true},
		{"Float64 equals", Eq("float64_val", 2.718), true},
		{"String num greater than", Gt("string_num", 100.0), true},
		{"Mixed type comparison", Gt("int_val", 40.5), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestRealWorldScenarios tests practical use cases
func TestRealWorldScenarios(t *testing.T) {
	products := []map[string]interface{}{
		{
			"id":       1,
			"name":     "iPhone 15 Pro",
			"category": "electronics",
			"price":    999.0,
			"stock":    50,
			"tags":     []string{"phone", "apple", "5g"},
			"specs": map[string]interface{}{
				"brand":  "Apple",
				"color":  "titanium",
				"storage": 256,
			},
		},
		{
			"id":       2,
			"name":     "Samsung Galaxy S24",
			"category": "electronics",
			"price":    899.0,
			"stock":    0,
			"tags":     []string{"phone", "samsung", "android"},
			"specs": map[string]interface{}{
				"brand":  "Samsung",
				"color":  "black",
				"storage": 128,
			},
		},
		{
			"id":       3,
			"name":     "MacBook Pro 16",
			"category": "computers",
			"price":    2499.0,
			"stock":    20,
			"tags":     []string{"laptop", "apple", "professional"},
			"specs": map[string]interface{}{
				"brand": "Apple",
				"cpu":   "M3 Pro",
				"ram":   32,
			},
		},
	}

	scenarios := []struct {
		name   string
		filter Filter
		want   []int // Expected product IDs
	}{
		{
			"Electronics under $1000",
			And(
				Eq("category", "electronics"),
				Lt("price", 1000.0),
			),
			[]int{1, 2},
		},
		{
			"Apple products in stock",
			And(
				Eq("specs.brand", "Apple"),
				Gt("stock", 0),
			),
			[]int{1, 3},
		},
		{
			"High-end products (>$2000) or out of stock",
			Or(
				Gt("price", 2000.0),
				Eq("stock", 0),
			),
			[]int{2, 3},
		},
		{
			"Phones with specific storage",
			And(
				Contains("tags", "phone"),
				Gte("specs.storage", 256),
			),
			[]int{1},
		},
		{
			"Not Samsung and in stock",
			And(
				Not(Eq("specs.brand", "Samsung")),
				Gt("stock", 0),
			),
			[]int{1, 3},
		},
		{
			"Products with 5g tag",
			Contains("tags", "5g"),
			[]int{1},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var matched []int
			for _, product := range products {
				if scenario.filter.Evaluate(product) {
					matched = append(matched, product["id"].(int))
				}
			}

			if len(matched) != len(scenario.want) {
				t.Errorf("%s: got %d matches, want %d", scenario.name, len(matched), len(scenario.want))
				t.Errorf("  Got IDs: %v", matched)
				t.Errorf("  Want IDs: %v", scenario.want)
				return
			}

			for i, id := range matched {
				if id != scenario.want[i] {
					t.Errorf("%s: got ID %d at position %d, want %d",
						scenario.name, id, i, scenario.want[i])
				}
			}
		})
	}
}

// TestJSONParsing tests filter creation from JSON
func TestJSONParsing(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		metadata map[string]interface{}
		want     bool
	}{
		{
			"Simple equality",
			`{"category": "electronics"}`,
			map[string]interface{}{"category": "electronics"},
			true,
		},
		{
			"Greater than",
			`{"price": {"$gt": 100}}`,
			map[string]interface{}{"price": 150.0},
			true,
		},
		{
			"AND operator",
			`{"$and": [{"category": "electronics"}, {"price": {"$lt": 1000}}]}`,
			map[string]interface{}{"category": "electronics", "price": 500.0},
			true,
		},
		{
			"OR operator",
			`{"$or": [{"category": "books"}, {"category": "electronics"}]}`,
			map[string]interface{}{"category": "electronics"},
			true,
		},
		{
			"NOT operator",
			`{"$not": {"category": "books"}}`,
			map[string]interface{}{"category": "electronics"},
			true,
		},
		{
			"IN operator",
			`{"category": {"$in": ["electronics", "computers"]}}`,
			map[string]interface{}{"category": "electronics"},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter, err := FromJSON([]byte(tt.json))
			if err != nil {
				t.Fatalf("Failed to parse JSON: %v", err)
			}

			if got := filter.Evaluate(tt.metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestFilterString tests string representation
func TestFilterString(t *testing.T) {
	tests := []struct {
		name   string
		filter Filter
		want   string
	}{
		{"Comparison", Eq("price", 100), "price eq 100"},
		{"And", And(Eq("a", 1), Eq("b", 2)), "AND(a eq 1, b eq 2)"},
		{"Or", Or(Eq("a", 1), Eq("b", 2)), "OR(a eq 1, b eq 2)"},
		{"Not", Not(Eq("a", 1)), "NOT(a eq 1)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.String(); got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGeoRadiusFilter tests geo radius filtering
func TestGeoRadiusFilter(t *testing.T) {
	// Times Square: 40.7580, -73.9855
	timesSquare := map[string]interface{}{
		"location": []interface{}{40.7580, -73.9855},
	}
	// Central Park (about 1.5km from Times Square)
	centralPark := map[string]interface{}{
		"location": map[string]interface{}{"lat": 40.7712, "lon": -73.9742},
	}
	// Brooklyn (about 8km from Times Square)
	brooklyn := map[string]interface{}{
		"location": []float64{40.6782, -73.9442},
	}
	noGeo := map[string]interface{}{
		"name": "no location",
	}

	tests := []struct {
		name     string
		filter   Filter
		metadata map[string]interface{}
		want     bool
	}{
		{"Inside radius (array format)", GeoRadius("location", 40.7580, -73.9855, 10), timesSquare, true},
		{"Inside radius (object format)", GeoRadius("location", 40.7580, -73.9855, 10), centralPark, true},
		{"Outside radius", GeoRadius("location", 40.7580, -73.9855, 5), brooklyn, false},
		{"Inside large radius (float64 array)", GeoRadius("location", 40.7580, -73.9855, 15), brooklyn, true},
		{"Missing geo field", GeoRadius("location", 40.7580, -73.9855, 10), noGeo, false},
		{"Zero radius (same point)", GeoRadius("location", 40.7580, -73.9855, 0), timesSquare, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(tt.metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestGeoBBoxFilter tests geo bounding box filtering
func TestGeoBBoxFilter(t *testing.T) {
	// Bounding box roughly covering Manhattan
	// Top-left: 40.80, -74.02   Bottom-right: 40.70, -73.93
	manhattanBBox := GeoBBox("location", 40.80, -74.02, 40.70, -73.93)

	tests := []struct {
		name     string
		filter   Filter
		metadata map[string]interface{}
		want     bool
	}{
		{
			"Inside bbox (array)",
			manhattanBBox,
			map[string]interface{}{"location": []interface{}{40.7580, -73.9855}},
			true,
		},
		{
			"Inside bbox (object)",
			manhattanBBox,
			map[string]interface{}{"location": map[string]interface{}{"lat": 40.7580, "lon": -73.9855}},
			true,
		},
		{
			"Outside bbox (too far south)",
			manhattanBBox,
			map[string]interface{}{"location": []interface{}{40.6500, -73.9500}},
			false,
		},
		{
			"Outside bbox (too far east)",
			manhattanBBox,
			map[string]interface{}{"location": []interface{}{40.7500, -73.9000}},
			false,
		},
		{
			"On boundary (top-left corner)",
			manhattanBBox,
			map[string]interface{}{"location": []interface{}{40.80, -74.02}},
			true,
		},
		{
			"On boundary (bottom-right corner)",
			manhattanBBox,
			map[string]interface{}{"location": []interface{}{40.70, -73.93}},
			true,
		},
		{
			"Missing field",
			manhattanBBox,
			map[string]interface{}{"name": "test"},
			false,
		},
		{
			"Float64 array format",
			manhattanBBox,
			map[string]interface{}{"location": []float64{40.75, -73.97}},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.filter.Evaluate(tt.metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestGeoFilterJSON tests JSON parsing of geo filters
func TestGeoFilterJSON(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		metadata map[string]interface{}
		want     bool
	}{
		{
			"Geo radius from JSON",
			`{"location": {"$geo_radius": {"lat": 40.7580, "lon": -73.9855, "radius_km": 10}}}`,
			map[string]interface{}{"location": []interface{}{40.7580, -73.9855}},
			true,
		},
		{
			"Geo radius outside",
			`{"location": {"$geo_radius": {"lat": 40.7580, "lon": -73.9855, "radius_km": 1}}}`,
			map[string]interface{}{"location": []interface{}{40.6782, -73.9442}},
			false,
		},
		{
			"Geo bbox from JSON",
			`{"location": {"$geo_bbox": {"top_left": [40.80, -74.02], "bottom_right": [40.70, -73.93]}}}`,
			map[string]interface{}{"location": []interface{}{40.75, -73.97}},
			true,
		},
		{
			"Geo bbox outside",
			`{"location": {"$geo_bbox": {"top_left": [40.80, -74.02], "bottom_right": [40.70, -73.93]}}}`,
			map[string]interface{}{"location": []interface{}{40.60, -73.97}},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter, err := FromJSON([]byte(tt.json))
			if err != nil {
				t.Fatalf("Failed to parse JSON: %v", err)
			}
			if got := filter.Evaluate(tt.metadata); got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

// TestGeoFilterCombined tests geo filters combined with other filters
func TestGeoFilterCombined(t *testing.T) {
	restaurants := []map[string]interface{}{
		{
			"name":     "Joe's Pizza",
			"category": "restaurant",
			"location": []interface{}{40.7308, -73.9973},
		},
		{
			"name":     "Shake Shack",
			"category": "restaurant",
			"location": []interface{}{40.7415, -73.9880},
		},
		{
			"name":     "Brooklyn Brewery",
			"category": "bar",
			"location": []interface{}{40.7215, -73.9573},
		},
		{
			"name":     "Far Away Place",
			"category": "restaurant",
			"location": []interface{}{41.0000, -74.5000},
		},
	}

	tests := []struct {
		name string
		filter Filter
		wantNames []string
	}{
		{
			"Restaurants within 5km of Union Square",
			And(
				GeoRadius("location", 40.7359, -73.9911, 5),
				Eq("category", "restaurant"),
			),
			[]string{"Joe's Pizza", "Shake Shack"},
		},
		{
			"Any place in bbox OR category bar",
			Or(
				GeoBBox("location", 40.75, -74.00, 40.73, -73.98),
				Eq("category", "bar"),
			),
			[]string{"Joe's Pizza", "Shake Shack", "Brooklyn Brewery"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var matched []string
			for _, r := range restaurants {
				if tt.filter.Evaluate(r) {
					matched = append(matched, r["name"].(string))
				}
			}
			if len(matched) != len(tt.wantNames) {
				t.Errorf("got %d matches %v, want %d %v", len(matched), matched, len(tt.wantNames), tt.wantNames)
				return
			}
			for i, name := range matched {
				if name != tt.wantNames[i] {
					t.Errorf("match[%d] = %q, want %q", i, name, tt.wantNames[i])
				}
			}
		})
	}
}

// BenchmarkFilterEvaluation benchmarks filter performance
func BenchmarkFilterEvaluation(b *testing.B) {
	metadata := map[string]interface{}{
		"price":    150.0,
		"category": "electronics",
		"stock":    10,
		"tags":     []string{"phone", "mobile", "5g"},
		"specs": map[string]interface{}{
			"brand":  "Apple",
			"storage": 256,
		},
	}

	benchmarks := []struct {
		name   string
		filter Filter
	}{
		{"Simple equality", Eq("category", "electronics")},
		{"Numeric comparison", Gt("price", 100.0)},
		{"Nested access", Eq("specs.brand", "Apple")},
		{"Array contains", Contains("tags", "phone")},
		{
			"Complex AND",
			And(
				Eq("category", "electronics"),
				Gt("price", 100.0),
				Lt("stock", 20),
			),
		},
		{
			"Complex nested",
			And(
				Eq("specs.brand", "Apple"),
				Gte("specs.storage", 128),
				Contains("tags", "phone"),
			),
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				bm.filter.Evaluate(metadata)
			}
		})
	}
}
