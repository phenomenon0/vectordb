package heap

import "container/heap"

// Lessable is an interface that allows a type to be compared to another of the same type.
// It is used to define the order of elements in the heap.
type Lessable[T any] interface {
	Less(T) bool
}

// innerHeap is a type that represents the heap data structure.
// it implements the std heap interface.
type innerHeap[T Lessable[T]] struct {
	data []T
}

func (h *innerHeap[T]) Len() int {
	return len(h.data)
}

func (h *innerHeap[T]) Less(i, j int) bool {
	return h.data[i].Less(h.data[j])
}

func (h *innerHeap[T]) Swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *innerHeap[T]) Push(x interface{}) {
	h.data = append(h.data, x.(T))
}

func (h *innerHeap[T]) Pop() interface{} {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[:n-1]
	return x
}

// Heap represents the heap data structure using a flat array to store the elements.
// It is a wrapper around the standard library's heap.
type Heap[T Lessable[T]] struct {
	inner innerHeap[T]
}

// Init establishes the heap invariants required by the other routines in this package.
// Init is idempotent with respect to the heap invariants
// and may be called whenever the heap invariants may have been invalidated.
// The complexity is O(n) where n = h.Len().
func (h *Heap[T]) Init(d []T) {
	h.inner.data = d
	heap.Init(&h.inner)
}

// Len returns the number of elements in the heap.
func (h *Heap[T]) Len() int {
	return h.inner.Len()
}

// Push pushes the element x onto the heap.
// The complexity is O(log n) where n = h.Len().
func (h *Heap[T]) Push(x T) {
	heap.Push(&h.inner, x)
}

// Pop removes and returns the minimum element (according to Less) from the heap.
// The complexity is O(log n) where n = h.Len().
// Pop is equivalent to Remove(h, 0).
func (h *Heap[T]) Pop() T {
	return heap.Pop(&h.inner).(T)
}

// PopLast removes and returns the maximum element from the min-heap.
func (h *Heap[T]) PopLast() T {
	return h.Remove(h.MaxIndex())
}

// Remove removes and returns the element at index i from the heap.
// The complexity is O(log n) where n = h.Len().
func (h *Heap[T]) Remove(i int) T {
	return heap.Remove(&h.inner, i).(T)
}

// Min returns the minimum element in the heap.
func (h *Heap[T]) Min() T {
	return h.inner.data[0]
}

// Max returns the maximum element in the min-heap.
// The maximum is among the leaf nodes (indices n/2 to n-1).
func (h *Heap[T]) Max() T {
	n := h.inner.Len()
	if n == 0 {
		var zero T
		return zero
	}
	if n == 1 {
		return h.inner.data[0]
	}
	// In a min-heap, the max is among the leaves: indices [n/2, n)
	maxIdx := n / 2
	for i := maxIdx + 1; i < n; i++ {
		if h.inner.data[maxIdx].Less(h.inner.data[i]) {
			maxIdx = i
		}
	}
	return h.inner.data[maxIdx]
}

// MaxIndex returns the index of the maximum element in the min-heap.
func (h *Heap[T]) MaxIndex() int {
	n := h.inner.Len()
	if n <= 1 {
		return 0
	}
	maxIdx := n / 2
	for i := maxIdx + 1; i < n; i++ {
		if h.inner.data[maxIdx].Less(h.inner.data[i]) {
			maxIdx = i
		}
	}
	return maxIdx
}

func (h *Heap[T]) Slice() []T {
	return h.inner.data
}
