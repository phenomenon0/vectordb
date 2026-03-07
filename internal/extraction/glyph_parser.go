package extraction

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// parseGlyphKnowledgeGraph parses a Glyph tabular response into a KnowledgeGraph.
// Falls back to JSON parsing if Glyph parsing fails (LLM may ignore Glyph instruction).
func parseGlyphKnowledgeGraph(response string) (*KnowledgeGraph, error) {
	response = strings.TrimSpace(response)

	// Strip markdown code blocks if the LLM wrapped the response
	response = cleanGlyphResponse(response)

	// Try Glyph parsing first
	kg, err := doParseGlyphKG(response)
	if err == nil && (len(kg.Nodes) > 0 || len(kg.Edges) > 0) {
		*kg = filterInvalidEdges(*kg)
		return kg, nil
	}

	// Fallback: try JSON parsing (LLM may have ignored Glyph instruction)
	cleaned := cleanJSONResponse(response)
	var jsonKG KnowledgeGraph
	if jsonErr := json.Unmarshal([]byte(cleaned), &jsonKG); jsonErr == nil {
		jsonKG = filterInvalidEdges(jsonKG)
		return &jsonKG, nil
	}

	// Both failed — return the Glyph error since that was the intended format
	if err != nil {
		return nil, fmt.Errorf("glyph parse failed: %w (response: %s)", err, truncate(response, 200))
	}
	return &KnowledgeGraph{}, nil
}

// parseGlyphTemporalKnowledgeGraph parses a Glyph tabular response into a TemporalKnowledgeGraph.
// Falls back to JSON parsing if Glyph parsing fails.
func parseGlyphTemporalKnowledgeGraph(response string) (*TemporalKnowledgeGraph, error) {
	response = strings.TrimSpace(response)
	response = cleanGlyphResponse(response)

	// Try Glyph parsing first
	tkg, err := doParseGlyphTemporalKG(response)
	if err == nil && (len(tkg.Nodes) > 0 || len(tkg.Edges) > 0 || len(tkg.Events) > 0) {
		tkg.KnowledgeGraph = filterInvalidEdges(tkg.KnowledgeGraph)
		return tkg, nil
	}

	// Fallback: try JSON parsing
	cleaned := cleanJSONResponse(response)
	var jsonTKG TemporalKnowledgeGraph
	if jsonErr := json.Unmarshal([]byte(cleaned), &jsonTKG); jsonErr == nil {
		jsonTKG.KnowledgeGraph = filterInvalidEdges(jsonTKG.KnowledgeGraph)
		return &jsonTKG, nil
	}

	if err != nil {
		return nil, fmt.Errorf("glyph temporal parse failed: %w (response: %s)", err, truncate(response, 200))
	}
	return &TemporalKnowledgeGraph{}, nil
}

// cleanGlyphResponse strips markdown code blocks that LLMs sometimes add.
func cleanGlyphResponse(s string) string {
	s = strings.TrimSpace(s)
	// Remove ```glyph, ```json, or ``` wrappers
	for _, prefix := range []string{"```glyph", "```json", "```text", "```"} {
		if strings.HasPrefix(s, prefix) {
			s = strings.TrimPrefix(s, prefix)
			if strings.HasSuffix(s, "```") {
				s = strings.TrimSuffix(s, "```")
			}
			s = strings.TrimSpace(s)
			break
		}
	}
	return s
}

// doParseGlyphKG does the actual Glyph parsing for KnowledgeGraph.
func doParseGlyphKG(response string) (*KnowledgeGraph, error) {
	sections := parseGlyphSections(response)

	kg := &KnowledgeGraph{}

	// Parse Node sections
	for _, sec := range sections {
		switch sec.name {
		case "Node":
			nodes, err := parseNodeRows(sec.rows, sec.columns)
			if err != nil {
				return nil, fmt.Errorf("parsing Node table: %w", err)
			}
			kg.Nodes = append(kg.Nodes, nodes...)
		case "Edge":
			edges, err := parseEdgeRows(sec.rows, sec.columns)
			if err != nil {
				return nil, fmt.Errorf("parsing Edge table: %w", err)
			}
			kg.Edges = append(kg.Edges, edges...)
		}
	}

	return kg, nil
}

// doParseGlyphTemporalKG does the actual Glyph parsing for TemporalKnowledgeGraph.
func doParseGlyphTemporalKG(response string) (*TemporalKnowledgeGraph, error) {
	sections := parseGlyphSections(response)

	tkg := &TemporalKnowledgeGraph{}

	for _, sec := range sections {
		switch sec.name {
		case "Node":
			nodes, err := parseNodeRows(sec.rows, sec.columns)
			if err != nil {
				return nil, fmt.Errorf("parsing Node table: %w", err)
			}
			tkg.Nodes = append(tkg.Nodes, nodes...)
		case "Edge":
			edges, err := parseEdgeRows(sec.rows, sec.columns)
			if err != nil {
				return nil, fmt.Errorf("parsing Edge table: %w", err)
			}
			tkg.Edges = append(tkg.Edges, edges...)
		case "Event":
			events, err := parseEventRows(sec.rows, sec.columns)
			if err != nil {
				return nil, fmt.Errorf("parsing Event table: %w", err)
			}
			tkg.Events = append(tkg.Events, events...)
		}
	}

	return tkg, nil
}

// glyphSection represents a parsed @tab ... @end section.
type glyphSection struct {
	name    string   // e.g. "Node", "Edge", "Event"
	columns []string // e.g. ["id", "name", "type", "desc"]
	rows    [][]string
}

// parseGlyphSections splits the response into @tab/@end sections.
func parseGlyphSections(response string) []glyphSection {
	var sections []glyphSection
	lines := strings.Split(response, "\n")

	var current *glyphSection
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		if strings.HasPrefix(trimmed, "@tab ") {
			// Parse header: @tab Name [col1 col2 col3]
			sec := parseTabHeader(trimmed)
			if sec != nil {
				sections = append(sections, *sec)
				current = &sections[len(sections)-1]
			}
			continue
		}

		if trimmed == "@end" {
			current = nil
			continue
		}

		if current != nil {
			fields := splitGlyphFields(trimmed)
			if len(fields) > 0 {
				current.rows = append(current.rows, fields)
			}
		}
	}

	return sections
}

// parseTabHeader parses "@tab Name [col1 col2 col3]" into a glyphSection.
func parseTabHeader(line string) *glyphSection {
	// Remove "@tab " prefix
	rest := strings.TrimPrefix(line, "@tab ")
	rest = strings.TrimSpace(rest)

	// Find section name (before the bracket)
	bracketIdx := strings.Index(rest, "[")
	if bracketIdx < 0 {
		// No columns specified, just the name
		return &glyphSection{name: strings.TrimSpace(rest)}
	}

	name := strings.TrimSpace(rest[:bracketIdx])

	// Extract columns from [col1 col2 col3]
	colPart := rest[bracketIdx:]
	colPart = strings.TrimPrefix(colPart, "[")
	colPart = strings.TrimSuffix(colPart, "]")
	colPart = strings.TrimSpace(colPart)

	var columns []string
	for _, col := range strings.Fields(colPart) {
		columns = append(columns, col)
	}

	return &glyphSection{
		name:    name,
		columns: columns,
	}
}

// splitGlyphFields splits a Glyph row into fields, respecting quoted strings
// and bracket-delimited lists.
// Examples:
//
//	entity_id "Entity Name" TYPE "Brief description"
//	source_id target_id RELATIONSHIP 1.0
//	event_id "What happened" 2024 "in 2024" [entity1 entity2]
func splitGlyphFields(line string) []string {
	var fields []string
	runes := []rune(line)
	i := 0
	n := len(runes)

	for i < n {
		// Skip whitespace
		for i < n && (runes[i] == ' ' || runes[i] == '\t') {
			i++
		}
		if i >= n {
			break
		}

		switch runes[i] {
		case '"':
			// Quoted string — find closing quote
			i++ // skip opening quote
			start := i
			for i < n && runes[i] != '"' {
				if runes[i] == '\\' && i+1 < n {
					i++ // skip escaped char
				}
				i++
			}
			fields = append(fields, string(runes[start:i]))
			if i < n {
				i++ // skip closing quote
			}

		case '[':
			// Bracket-delimited list — capture the whole thing including brackets
			start := i
			depth := 0
			for i < n {
				if runes[i] == '[' {
					depth++
				} else if runes[i] == ']' {
					depth--
					if depth == 0 {
						i++
						break
					}
				}
				i++
			}
			fields = append(fields, string(runes[start:i]))

		default:
			// Unquoted token
			start := i
			for i < n && runes[i] != ' ' && runes[i] != '\t' {
				i++
			}
			fields = append(fields, string(runes[start:i]))
		}
	}

	return fields
}

// parseNodeRows converts parsed row fields into Node structs.
// Expected columns: [id name type desc]
func parseNodeRows(rows [][]string, columns []string) ([]Node, error) {
	colIdx := buildColumnIndex(columns)
	var nodes []Node

	for _, fields := range rows {
		if len(fields) == 0 {
			continue
		}

		node := Node{}

		// Map by column index, falling back to positional
		if idx, ok := colIdx["id"]; ok && idx < len(fields) {
			node.ID = fields[idx]
		} else if len(fields) > 0 {
			node.ID = fields[0]
		}

		if idx, ok := colIdx["name"]; ok && idx < len(fields) {
			node.Name = fields[idx]
		} else if len(fields) > 1 {
			node.Name = fields[1]
		}

		if idx, ok := colIdx["type"]; ok && idx < len(fields) {
			node.Type = fields[idx]
		} else if len(fields) > 2 {
			node.Type = fields[2]
		}

		if idx, ok := colIdx["desc"]; ok && idx < len(fields) {
			node.Description = fields[idx]
		} else if idx, ok := colIdx["description"]; ok && idx < len(fields) {
			node.Description = fields[idx]
		} else if len(fields) > 3 {
			node.Description = fields[3]
		}

		// If name is empty, use ID
		if node.Name == "" {
			node.Name = node.ID
		}

		if node.ID != "" && node.Type != "" {
			nodes = append(nodes, node)
		}
	}

	return nodes, nil
}

// parseEdgeRows converts parsed row fields into Edge structs.
// Expected columns: [src tgt rel w]
func parseEdgeRows(rows [][]string, columns []string) ([]Edge, error) {
	colIdx := buildColumnIndex(columns)
	var edges []Edge

	for _, fields := range rows {
		if len(fields) == 0 {
			continue
		}

		edge := Edge{
			Weight: 1.0, // default weight
		}

		if idx, ok := colIdx["src"]; ok && idx < len(fields) {
			edge.SourceNodeID = fields[idx]
		} else if idx, ok := colIdx["source"]; ok && idx < len(fields) {
			edge.SourceNodeID = fields[idx]
		} else if len(fields) > 0 {
			edge.SourceNodeID = fields[0]
		}

		if idx, ok := colIdx["tgt"]; ok && idx < len(fields) {
			edge.TargetNodeID = fields[idx]
		} else if idx, ok := colIdx["target"]; ok && idx < len(fields) {
			edge.TargetNodeID = fields[idx]
		} else if len(fields) > 1 {
			edge.TargetNodeID = fields[1]
		}

		if idx, ok := colIdx["rel"]; ok && idx < len(fields) {
			edge.RelationshipName = fields[idx]
		} else if idx, ok := colIdx["relationship"]; ok && idx < len(fields) {
			edge.RelationshipName = fields[idx]
		} else if len(fields) > 2 {
			edge.RelationshipName = fields[2]
		}

		if idx, ok := colIdx["w"]; ok && idx < len(fields) {
			if w, err := strconv.ParseFloat(fields[idx], 32); err == nil {
				edge.Weight = float32(w)
			}
		} else if idx, ok := colIdx["weight"]; ok && idx < len(fields) {
			if w, err := strconv.ParseFloat(fields[idx], 32); err == nil {
				edge.Weight = float32(w)
			}
		} else if len(fields) > 3 {
			if w, err := strconv.ParseFloat(fields[3], 32); err == nil {
				edge.Weight = float32(w)
			}
		}

		if edge.SourceNodeID != "" && edge.TargetNodeID != "" && edge.RelationshipName != "" {
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// parseEventRows converts parsed row fields into TemporalEvent structs.
// Expected columns: [id desc year date_text entities]
func parseEventRows(rows [][]string, columns []string) ([]TemporalEvent, error) {
	colIdx := buildColumnIndex(columns)
	var events []TemporalEvent

	for _, fields := range rows {
		if len(fields) == 0 {
			continue
		}

		event := TemporalEvent{}

		if idx, ok := colIdx["id"]; ok && idx < len(fields) {
			event.ID = fields[idx]
		} else if len(fields) > 0 {
			event.ID = fields[0]
		}

		if idx, ok := colIdx["desc"]; ok && idx < len(fields) {
			event.Description = fields[idx]
		} else if idx, ok := colIdx["description"]; ok && idx < len(fields) {
			event.Description = fields[idx]
		} else if len(fields) > 1 {
			event.Description = fields[1]
		}

		if idx, ok := colIdx["year"]; ok && idx < len(fields) {
			if y, err := strconv.Atoi(fields[idx]); err == nil {
				event.Year = y
			}
		} else if len(fields) > 2 {
			if y, err := strconv.Atoi(fields[2]); err == nil {
				event.Year = y
			}
		}

		if idx, ok := colIdx["date_text"]; ok && idx < len(fields) {
			event.DateText = fields[idx]
		} else if len(fields) > 3 {
			event.DateText = fields[3]
		}

		if idx, ok := colIdx["entities"]; ok && idx < len(fields) {
			event.Entities = parseBracketList(fields[idx])
		} else if len(fields) > 4 {
			event.Entities = parseBracketList(fields[4])
		}

		if event.ID != "" {
			events = append(events, event)
		}
	}

	return events, nil
}

// buildColumnIndex creates a map from column name to index.
func buildColumnIndex(columns []string) map[string]int {
	idx := make(map[string]int, len(columns))
	for i, col := range columns {
		idx[strings.ToLower(col)] = i
	}
	return idx
}

// parseBracketList parses "[item1 item2 item3]" into a string slice.
func parseBracketList(s string) []string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	s = strings.TrimSpace(s)

	if s == "" {
		return nil
	}

	var items []string
	for _, item := range strings.Fields(s) {
		item = strings.TrimSpace(item)
		if item != "" {
			items = append(items, item)
		}
	}
	return items
}
