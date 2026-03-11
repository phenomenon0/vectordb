// Package hnsw is a fork of github.com/coder/hnsw with Windows compatibility.
// The SavedGraph feature using renameio has been removed since we only use
// Export/Import with io.Reader/io.Writer.
package hnsw

import (
	"encoding/binary"
	"fmt"
	"io"
)

var byteOrder = binary.LittleEndian

func binaryRead(r io.Reader, data interface{}) (int, error) {
	switch v := data.(type) {
	case *int:
		br, ok := r.(io.ByteReader)
		if !ok {
			return 0, fmt.Errorf("reader does not implement io.ByteReader")
		}

		i, err := binary.ReadVarint(br)
		if err != nil {
			return 0, err
		}

		*v = int(i)
		return binary.MaxVarintLen64, nil

	case *string:
		var ln int
		_, err := binaryRead(r, &ln)
		if err != nil {
			return 0, err
		}

		s := make([]byte, ln)
		_, err = binaryRead(r, &s)
		*v = string(s)
		return len(s), err

	case *[]float32:
		var ln int
		_, err := binaryRead(r, &ln)
		if err != nil {
			return 0, err
		}

		*v = make([]float32, ln)
		return binary.Size(*v), binary.Read(r, byteOrder, *v)

	case io.ReaderFrom:
		n, err := v.ReadFrom(r)
		return int(n), err

	default:
		return binary.Size(data), binary.Read(r, byteOrder, data)
	}
}

func binaryWrite(w io.Writer, data any) (int, error) {
	switch v := data.(type) {
	case int:
		var buf [binary.MaxVarintLen64]byte
		n := binary.PutVarint(buf[:], int64(v))
		n, err := w.Write(buf[:n])
		return n, err
	case io.WriterTo:
		n, err := v.WriteTo(w)
		return int(n), err
	case string:
		n, err := binaryWrite(w, len(v))
		if err != nil {
			return n, err
		}
		n2, err := io.WriteString(w, v)
		if err != nil {
			return n + n2, err
		}

		return n + n2, nil
	case []float32:
		n, err := binaryWrite(w, len(v))
		if err != nil {
			return n, err
		}
		return n + binary.Size(v), binary.Write(w, byteOrder, v)

	default:
		sz := binary.Size(data)
		err := binary.Write(w, byteOrder, data)
		if err != nil {
			return 0, fmt.Errorf("encoding %T: %w", data, err)
		}
		return sz, err
	}
}

func multiBinaryWrite(w io.Writer, data ...any) (int, error) {
	var written int
	for _, d := range data {
		n, err := binaryWrite(w, d)
		written += n
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

func multiBinaryRead(r io.Reader, data ...any) (int, error) {
	var read int
	for i, d := range data {
		n, err := binaryRead(r, d)
		read += n
		if err != nil {
			return read, fmt.Errorf("reading %T at index %v: %w", d, i, err)
		}
	}
	return read, nil
}

const encodingVersion = 1

// Export writes the graph to a writer.
// T must implement io.WriterTo.
func (h *Graph[K]) Export(w io.Writer) error {
	distFuncName, ok := distanceFuncToName(h.Distance)
	if !ok {
		return fmt.Errorf("distance function %v must be registered with RegisterDistanceFunc", h.Distance)
	}
	_, err := multiBinaryWrite(
		w,
		encodingVersion,
		h.M,
		h.Ml,
		h.EfSearch,
		distFuncName,
	)
	if err != nil {
		return fmt.Errorf("encode parameters: %w", err)
	}
	_, err = binaryWrite(w, len(h.layers))
	if err != nil {
		return fmt.Errorf("encode number of layers: %w", err)
	}
	for _, layer := range h.layers {
		_, err = binaryWrite(w, len(layer.nodes))
		if err != nil {
			return fmt.Errorf("encode number of nodes: %w", err)
		}
		for _, node := range layer.nodes {
			nb := node.loadNeighbors()
			_, err = multiBinaryWrite(w, node.Key, node.Value, len(nb))
			if err != nil {
				return fmt.Errorf("encode node data: %w", err)
			}

			for _, neighbor := range nb {
				_, err = binaryWrite(w, neighbor.Key)
				if err != nil {
					return fmt.Errorf("encode neighbor %v: %w", neighbor.Key, err)
				}
			}
		}
	}

	return nil
}

// Import reads the graph from a reader.
// T must implement io.ReaderFrom.
// The imported graph does not have to match the exported graph's parameters (except for
// dimensionality). The graph will converge onto the new parameters.
func (h *Graph[K]) Import(r io.Reader) error {
	var (
		version int
		dist    string
	)
	_, err := multiBinaryRead(r, &version, &h.M, &h.Ml, &h.EfSearch,
		&dist,
	)
	if err != nil {
		return err
	}

	var ok bool
	h.Distance, ok = distanceFuncs[dist]
	if !ok {
		return fmt.Errorf("unknown distance function %q", dist)
	}
	if h.Rng == nil {
		h.Rng = defaultRand()
	}

	if version != encodingVersion {
		return fmt.Errorf("incompatible encoding version: %d", version)
	}

	var nLayers int
	_, err = binaryRead(r, &nLayers)
	if err != nil {
		return err
	}

	h.layers = make([]*layer[K], nLayers)
	for i := 0; i < nLayers; i++ {
		var nNodes int
		_, err = binaryRead(r, &nNodes)
		if err != nil {
			return err
		}

		// First pass: create all nodes and read neighbor keys
		nodes := make(map[K]*layerNode[K], nNodes)
		allNodes := make([]*layerNode[K], 0, nNodes)
		allNeighborKeys := make([][]K, 0, nNodes)

		for j := 0; j < nNodes; j++ {
			var key K
			var vec Vector
			var nNeighbors int
			_, err = multiBinaryRead(r, &key, &vec, &nNeighbors)
			if err != nil {
				return fmt.Errorf("decoding node %d: %w", j, err)
			}

			neighborKeys := make([]K, nNeighbors)
			for k := 0; k < nNeighbors; k++ {
				_, err = binaryRead(r, &neighborKeys[k])
				if err != nil {
					return fmt.Errorf("decoding neighbor %d for node %d: %w", k, j, err)
				}
			}

			node := &layerNode[K]{
				Node: Node[K]{
					Key:   key,
					Value: vec,
				},
			}

			nodes[key] = node
			allNodes = append(allNodes, node)
			allNeighborKeys = append(allNeighborKeys, neighborKeys)
		}

		// Second pass: resolve neighbor keys to pointers
		for idx, node := range allNodes {
			keys := allNeighborKeys[idx]
			nb := make([]*layerNode[K], 0, len(keys))
			for _, key := range keys {
				if neighbor, ok := nodes[key]; ok {
					nb = append(nb, neighbor)
				}
			}
			node.storeNeighbors(nb)
		}

		h.layers[i] = &layer[K]{nodes: nodes}
	}

	return nil
}
