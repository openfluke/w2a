package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/stub/seed"
	"github.com/openfluke/welvet/weights"
)

//export WelvetListDTypes
func WelvetListDTypes() *C.char {
	out := make([]map[string]any, 0, len(core.AllDTypes))
	for _, d := range core.AllDTypes {
		out = append(out, map[string]any{"id": int(d), "name": d.String()})
	}
	return jsonOut(out)
}

//export WelvetListFormats
func WelvetListFormats() *C.char {
	out := make([]map[string]any, 0, len(quant.AllFormats))
	for _, f := range quant.AllFormats {
		out = append(out, map[string]any{"id": int(f), "name": f.String()})
	}
	return jsonOut(out)
}

//export WelvetListLayerTypes
func WelvetListLayerTypes() *C.char {
	return jsonOut([]string{
		"Dense", "MultiHeadAttention", "SwiGLU", "RMSNorm", "LayerNorm",
		"Embedding", "Softmax", "Sequential", "Residual",
		"CNN1", "CNN2", "CNN3", "RNN", "LSTM",
		"ConvT1", "ConvT2", "ConvT3", "Parallel", "Stack",
		"KMeans", "Mamba", "Metacognition", "GDN",
	})
}

//export WelvetListBackends
func WelvetListBackends() *C.char {
	return jsonOut([]string{"cpu_tiled", "simd", "webgpu"})
}

//export WelvetSeedFrom
func WelvetSeedFrom(partsJSON *C.char) *C.char {
	var parts []any
	if err := json.Unmarshal([]byte(C.GoString(partsJSON)), &parts); err != nil {
		return errJSON(err.Error())
	}
	v := seed.From(parts...)
	return C.CString(fmt.Sprintf("%d", v))
}

//export WelvetCreateStore
func WelvetCreateStore(rows, cols, dtype, format C.int, dataPtr *C.float, dataLen C.int) C.longlong {
	var init []float32
	if dataPtr != nil && dataLen > 0 {
		init = readF32(dataPtr, int(dataLen))
	}
	s, err := weights.New[float32](int(rows), int(cols), init, core.DType(dtype), quant.Format(format))
	if err != nil {
		return 0
	}
	return C.longlong(storeStore(s))
}

//export WelvetStoreApplySGD
func WelvetStoreApplySGD(handle C.longlong, dWPtr *C.double, dWLen C.int, lr C.double) *C.char {
	s, ok := getStore(int64(handle))
	if !ok {
		return errJSON("invalid store handle")
	}
	n := int(dWLen)
	dW := make([]float64, n)
	if dWPtr != nil && n > 0 {
		src := unsafeSliceF64(dWPtr, n)
		copy(dW, src)
	}
	if err := s.ApplySGD(dW, float64(lr)); err != nil {
		return errJSON(err.Error())
	}
	return okJSON()
}

//export WelvetStoreFlattenF32
func WelvetStoreFlattenF32(handle C.longlong, outPtr *C.float, outCap C.int, outLen *C.int) *C.char {
	s, ok := getStore(int64(handle))
	if !ok {
		return errJSON("invalid store handle")
	}
	flat, err := s.FlattenF32()
	if err != nil {
		return errJSON(err.Error())
	}
	if int(outCap) < len(flat) {
		return errJSON("outCap too small")
	}
	writeF32(outPtr, flat)
	if outLen != nil {
		*outLen = C.int(len(flat))
	}
	return okJSON()
}

//export WelvetFreeStore
func WelvetFreeStore(handle C.longlong) {
	mu.Lock()
	defer mu.Unlock()
	delete(stores, int64(handle))
}

func unsafeSliceF64(ptr *C.double, n int) []float64 {
	if ptr == nil || n <= 0 {
		return nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
}
