//go:build js && wasm

// Package main is the Welvet 1.1.1 WASM entrypoint (syscall/js over github.com/openfluke/welvet).
// Engine version must match @openfluke/welvet WELVET_ENGINE_VERSION.
package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"syscall/js"

	"github.com/openfluke/welvet/architecture"
)

// welvetEngineVersion must match typescript WELVET_ENGINE_VERSION.
const welvetEngineVersion = "1.1.1"

var (
	grids      = make(map[int64]*architecture.Grid)
	gridNextID int64 = 1

	stacks      = make(map[int64]any) // *parallel.Stack
	stackNextID int64 = 1

	parallels      = make(map[int64]any) // *parallel.Layer
	parallelNextID int64 = 1

	stepStates  = make(map[int64]any)
	stepNextID  int64 = 1

	tweenStates = make(map[int64]any)
	tweenNextID int64 = 1

	mu sync.RWMutex
)

func storeGrid(g *architecture.Grid) int64 {
	mu.Lock()
	id := gridNextID
	gridNextID++
	grids[id] = g
	mu.Unlock()
	return id
}

func getGrid(id int64) (*architecture.Grid, bool) {
	mu.RLock()
	defer mu.RUnlock()
	g, ok := grids[id]
	return g, ok
}

func freeGrid(id int64) {
	mu.Lock()
	delete(grids, id)
	mu.Unlock()
}

func storeStack(s any) int64 {
	mu.Lock()
	id := stackNextID
	stackNextID++
	stacks[id] = s
	mu.Unlock()
	return id
}

func getStack(id int64) (any, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := stacks[id]
	return s, ok
}

func storeParallel(p any) int64 {
	mu.Lock()
	id := parallelNextID
	parallelNextID++
	parallels[id] = p
	mu.Unlock()
	return id
}

func getParallel(id int64) (any, bool) {
	mu.RLock()
	defer mu.RUnlock()
	p, ok := parallels[id]
	return p, ok
}

func storeStep(s any) int64 {
	mu.Lock()
	id := stepNextID
	stepNextID++
	stepStates[id] = s
	mu.Unlock()
	return id
}

func getStep(id int64) (any, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := stepStates[id]
	return s, ok
}

func storeTween(s any) int64 {
	mu.Lock()
	id := tweenNextID
	tweenNextID++
	tweenStates[id] = s
	mu.Unlock()
	return id
}

func getTween(id int64) (any, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := tweenStates[id]
	return s, ok
}

func errObj(msg string) js.Value {
	obj := js.Global().Get("Object").New()
	obj.Set("error", msg)
	return obj
}

func okObj() js.Value {
	obj := js.Global().Get("Object").New()
	obj.Set("status", "ok")
	return obj
}

func jsFloat32Array(data []float32) js.Value {
	arr := js.Global().Get("Float32Array").New(len(data))
	for i, v := range data {
		arr.SetIndex(i, float64(v))
	}
	return arr
}

func readFloat32Array(jsVal js.Value) []float32 {
	if jsVal.IsUndefined() || jsVal.IsNull() {
		return nil
	}
	length := jsVal.Get("length").Int()
	out := make([]float32, length)
	for i := 0; i < length; i++ {
		out[i] = float32(jsVal.Index(i).Float())
	}
	return out
}

func readIntSlice(jsVal js.Value) []int {
	if jsVal.IsUndefined() || jsVal.IsNull() {
		return nil
	}
	if jsVal.Type() == js.TypeString {
		var out []int
		_ = json.Unmarshal([]byte(jsVal.String()), &out)
		return out
	}
	length := jsVal.Get("length").Int()
	out := make([]int, length)
	for i := 0; i < length; i++ {
		out[i] = jsVal.Index(i).Int()
	}
	return out
}

func jsUint8Array(data []byte) js.Value {
	arr := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(arr, data)
	return arr
}

func readUint8Array(jsVal js.Value) []byte {
	length := jsVal.Get("length").Int()
	out := make([]byte, length)
	js.CopyBytesToGo(out, jsVal)
	return out
}

func jsonStr(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"error":%q}`, err.Error())
	}
	return string(b)
}

func main() {
	registerGlobals()
	registerGridGlobals()
	registerLayerGlobals()
	registerRuntimeGlobals()
	registerParallelGlobals()
	registerSystemsGlobals()
	registerModelGlobals()
	registerLucyGlobals()
	registerStubGlobals()
	registerCameralPolyGlobals()

	js.Global().Set("welvetEngineVersion", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return welvetEngineVersion
	}))
	js.Global().Set("loomEngineVersion", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return welvetEngineVersion // Loom-compat alias
	}))
	js.Global().Set("welvetGC", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		runtime.GC()
		return okObj()
	}))
	js.Global().Set("getWelvetInternalParity", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr(totalParityItems)
	}))

	select {}
}

func registerGlobals() {
	// placeholder — other files register their own
}
