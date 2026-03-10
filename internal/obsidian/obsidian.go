// Package obsidian provides Obsidian vault parsing, note collection, backlink
// computation, and incremental sync state management. Used by both the CLI
// import-obsidian command and the server's background auto-sync.
package obsidian

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Note represents a single parsed markdown note from an Obsidian vault.
type Note struct {
	ID            string
	Content       string
	Meta          map[string]string
	OutgoingLinks []string
	FileSize      int64
	ModTime       time.Time
}

// SyncState tracks file mtimes for incremental sync.
type SyncState struct {
	Mtimes map[string]time.Time `json:"mtimes"`
}

var (
	ReWikiLink  = regexp.MustCompile(`\[\[([^\]|]+)(?:\|[^\]]+)?\]\]`)
	ReInlineTag = regexp.MustCompile(`(?:^|\s)#([\w\-/]+)`)
	ReZettelID  = regexp.MustCompile(`^\d{12,14}`)
)

// NoteName returns the bare note name (filename without .md extension).
func NoteName(id string) string {
	return strings.TrimSuffix(filepath.Base(id), ".md")
}

// LoadSyncState loads incremental sync state from disk.
func LoadSyncState(path string) SyncState {
	var s SyncState
	s.Mtimes = make(map[string]time.Time)
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	json.Unmarshal(data, &s)
	if s.Mtimes == nil {
		s.Mtimes = make(map[string]time.Time)
	}
	return s
}

// SaveSyncState persists incremental sync state to disk.
func SaveSyncState(path string, s SyncState) error {
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// CollectDirect walks vaultPath reading .md files directly from the filesystem.
func CollectDirect(vaultPath, vaultName, source string, excludeDirs map[string]bool, stripFM bool) ([]Note, error) {
	var notes []Note
	err := filepath.WalkDir(vaultPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable
		}
		if d.IsDir() {
			if excludeDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}
		// Skip macOS resource fork shadow files
		if strings.HasPrefix(filepath.Base(path), "._") {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.Size() == 0 || info.Size() > 1<<20 {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(vaultPath, path)
		note := ParseNote(rel, string(data), vaultName, source, info.ModTime(), info.Size(), stripFM)
		note.Meta["vault_path"] = path
		notes = append(notes, note)
		return nil
	})
	return notes, err
}

// ParseNote extracts metadata, tags, wiki-links, and zettelkasten IDs from a
// markdown note's content and path.
func ParseNote(relPath, content, vaultName, source string, modTime time.Time, fileSize int64, stripFM bool) Note {
	meta := map[string]string{
		"source": source,
		"vault":  vaultName,
		"path":   relPath,
		"folder": filepath.Dir(relPath),
	}

	baseName := NoteName(relPath)
	meta["title"] = baseName

	// Parse YAML frontmatter (single pass)
	body := content
	var fmTags []string
	if strings.HasPrefix(content, "---\n") {
		endIdx := strings.Index(content[4:], "\n---")
		if endIdx >= 0 {
			fmBlock := content[4 : 4+endIdx]
			body = content[4+endIdx+4:] // after closing ---\n

			inTags := false
			for _, line := range strings.Split(fmBlock, "\n") {
				trimmed := strings.TrimSpace(line)
				if trimmed == "" || strings.HasPrefix(trimmed, "#") {
					continue
				}

				// Handle multi-line tag list items
				if inTags {
					if strings.HasPrefix(trimmed, "- ") {
						t := strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))
						t = strings.TrimPrefix(t, "#")
						if t != "" {
							fmTags = append(fmTags, t)
						}
						continue
					}
					inTags = false
				}

				k, v, ok := strings.Cut(trimmed, ":")
				if !ok {
					continue
				}
				k = strings.TrimSpace(k)
				v = strings.TrimSpace(v)
				switch k {
				case "title":
					meta["title"] = strings.Trim(v, "\"'")
				case "author":
					meta["author"] = strings.NewReplacer("[[", "", "]]", "").Replace(strings.Trim(v, "\"'"))
				case "created", "date":
					meta["created"] = strings.Trim(v, "\"'")
				case "zettelkasten_id", "uid", "zettel":
					meta["zettelkasten_id"] = strings.Trim(v, "\"'")
				case "tags":
					// tags can be: [a, b, c] or a, b, c (inline) or empty (multi-line follows)
					v = strings.Trim(v, "[]\"'")
					if v == "" {
						inTags = true // next lines are "- tag" items
					} else {
						for _, t := range strings.Split(v, ",") {
							t = strings.TrimSpace(t)
							t = strings.TrimPrefix(t, "#")
							if t != "" {
								fmTags = append(fmTags, t)
							}
						}
					}
				}
			}
		}
	}

	// Extract inline #tags from body (skip headings: lines starting with #)
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") && (len(trimmed) < 2 || trimmed[1] == ' ' || trimmed[1] == '#') {
			continue // heading
		}
		for _, m := range ReInlineTag.FindAllStringSubmatch(line, -1) {
			fmTags = append(fmTags, m[1])
		}
	}

	// Deduplicate tags
	if len(fmTags) > 0 {
		seen := make(map[string]bool)
		var unique []string
		for _, t := range fmTags {
			lower := strings.ToLower(t)
			if !seen[lower] {
				seen[lower] = true
				unique = append(unique, t)
			}
		}
		meta["tags"] = strings.Join(unique, ",")
	}

	// Extract [[wiki-links]]
	var outgoing []string
	seen := make(map[string]bool)
	for _, m := range ReWikiLink.FindAllStringSubmatch(body, -1) {
		link := strings.TrimSpace(m[1])
		// Strip path prefix if present (e.g., folder/note -> note)
		if idx := strings.LastIndex(link, "/"); idx >= 0 {
			link = link[idx+1:]
		}
		if !seen[link] {
			seen[link] = true
			outgoing = append(outgoing, link)
		}
	}
	if len(outgoing) > 0 {
		meta["outgoing_links"] = strings.Join(outgoing, ",")
	}

	// Detect Zettelkasten ID from filename if not in frontmatter
	if meta["zettelkasten_id"] == "" {
		if m := ReZettelID.FindString(baseName); m != "" {
			meta["zettelkasten_id"] = m
		}
	}

	// Created from modTime if not in frontmatter
	if meta["created"] == "" && !modTime.IsZero() {
		meta["created"] = modTime.Format("2006-01-02")
	}

	// Optionally strip frontmatter from content
	docContent := content
	if stripFM {
		docContent = body
	}

	return Note{
		ID:            relPath,
		Content:       docContent,
		Meta:          meta,
		OutgoingLinks: outgoing,
		FileSize:      fileSize,
		ModTime:       modTime,
	}
}

// ComputeBacklinks builds reverse link maps and populates backlink metadata
// on each note. Must be called on the full set of notes.
func ComputeBacklinks(notes []Note) {
	// Build reverse map: link target name -> list of source note names
	backlinks := map[string][]string{}
	for _, note := range notes {
		name := NoteName(note.ID)
		for _, link := range note.OutgoingLinks {
			backlinks[link] = append(backlinks[link], name)
		}
	}
	// Apply to each note
	for i, note := range notes {
		name := NoteName(note.ID)
		if bl := backlinks[name]; len(bl) > 0 {
			notes[i].Meta["backlinks"] = strings.Join(bl, ",")
		}
	}
}

// DefaultExcludes returns the default set of directories to skip when walking a vault.
func DefaultExcludes() map[string]bool {
	return map[string]bool{
		".obsidian": true,
		".git":      true,
		".trash":    true,
		".space":    true,
		".views":    true,
	}
}
