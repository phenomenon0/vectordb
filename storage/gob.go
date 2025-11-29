package storage

import (
	"encoding/gob"
	"io"
)

// GobFormat implements Format using Go's gob encoding.
// This is the default format for backward compatibility.
type GobFormat struct{}

func init() {
	Register(&GobFormat{})
}

func (g *GobFormat) Name() string {
	return "gob"
}

func (g *GobFormat) Extension() string {
	return ".gob"
}

func (g *GobFormat) Save(w io.Writer, p *Payload) error {
	return gob.NewEncoder(w).Encode(p)
}

func (g *GobFormat) Load(r io.Reader) (*Payload, error) {
	var p Payload
	if err := gob.NewDecoder(r).Decode(&p); err != nil {
		return nil, err
	}
	return &p, nil
}
