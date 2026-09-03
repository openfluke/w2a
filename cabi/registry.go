package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/weights"
)

var (
	mu       sync.RWMutex
	grids    = map[int64]*architecture.Grid{}
	gridNext int64 = 1

	stores    = map[int64]*weights.Store{}
	storeNext int64 = 1

	stacks    = map[int64]*parallel.Stack{}
	stackNext int64 = 1

	parallels    = map[int64]*parallel.Layer{}
	parallelNext int64 = 1
)

func errJSON(msg string) *C.char {
	b, _ := json.Marshal(map[string]string{"error": msg})
	return C.CString(string(b))
}

func okJSON() *C.char {
	return C.CString(`{"status":"ok"}`)
}

func jsonOut(v any) *C.char {
	b, err := json.Marshal(v)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(b))
}

func storeGrid(g *architecture.Grid) int64 {
	mu.Lock()
	defer mu.Unlock()
	id := gridNext
	gridNext++
	grids[id] = g
	return id
}

func getGrid(id int64) (*architecture.Grid, bool) {
	mu.RLock()
	defer mu.RUnlock()
	g, ok := grids[id]
	return g, ok
}

func deleteGrid(id int64) {
	mu.Lock()
	defer mu.Unlock()
	delete(grids, id)
}

func storeStore(s *weights.Store) int64 {
	mu.Lock()
	defer mu.Unlock()
	id := storeNext
	storeNext++
	stores[id] = s
	return id
}

func getStore(id int64) (*weights.Store, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := stores[id]
	return s, ok
}

func storeStack(s *parallel.Stack) int64 {
	mu.Lock()
	defer mu.Unlock()
	id := stackNext
	stackNext++
	stacks[id] = s
	return id
}

func getStack(id int64) (*parallel.Stack, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := stacks[id]
	return s, ok
}

func storeParallel(p *parallel.Layer) int64 {
	mu.Lock()
	defer mu.Unlock()
	id := parallelNext
	parallelNext++
	parallels[id] = p
	return id
}

func getParallel(id int64) (*parallel.Layer, bool) {
	mu.RLock()
	defer mu.RUnlock()
	p, ok := parallels[id]
	return p, ok
}

func readF32(ptr *C.float, n int) []float32 {
	if ptr == nil || n <= 0 {
		return nil
	}
	out := make([]float32, n)
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
	copy(out, src)
	return out
}

func writeF32(ptr *C.float, src []float32) {
	if ptr == nil || len(src) == 0 {
		return
	}
	dst := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), len(src))
	copy(dst, src)
}

func unsafeSliceU8(ptr *C.uchar, n int) []byte {
	if ptr == nil || n <= 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n)
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"error":%q}`, err.Error())
	}
	return string(b)
}
