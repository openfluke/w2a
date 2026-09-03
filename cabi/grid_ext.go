package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/weights"
)

type gridSpec struct {
	Depth         int    `json:"depth"`
	Rows          int    `json:"rows"`
	Cols          int    `json:"cols"`
	LayersPerCell int    `json:"layers_per_cell"`
	Backend       string `json:"backend,omitempty"`
}

type placeSpec struct {
	Z      int    `json:"z"`
	Y      int    `json:"y"`
	X      int    `json:"x"`
	L      int    `json:"l"`
	In     int    `json:"in"`
	Out    int    `json:"out"`
	Dim    int    `json:"dim"`
	Act    string `json:"act"`
	DType  int    `json:"dtype"`
	Format int    `json:"format"`
}

//export WelvetCreateGrid
func WelvetCreateGrid(cfgJSON *C.char) C.longlong {
	raw := C.GoString(cfgJSON)
	var spec gridSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return 0
	}
	if spec.Depth < 1 {
		spec.Depth = 1
	}
	if spec.Rows < 1 {
		spec.Rows = 1
	}
	if spec.Cols < 1 {
		spec.Cols = 1
	}
	if spec.LayersPerCell < 1 {
		spec.LayersPerCell = 1
	}
	g := architecture.NewGrid(spec.Depth, spec.Rows, spec.Cols, spec.LayersPerCell)
	switch spec.Backend {
	case "simd", "SIMD":
		g.Exec.Backend = core.BackendSIMD
	case "webgpu", "WebGPU":
		g.Exec.Backend = core.BackendWebGPU
	default:
		g.Exec.Backend = core.BackendCPUTiled
	}
	return C.longlong(storeGrid(g))
}

//export WelvetFreeGrid
func WelvetFreeGrid(handle C.longlong) {
	deleteGrid(int64(handle))
}

//export WelvetGridInfo
func WelvetGridInfo(handle C.longlong) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	return jsonOut(map[string]any{
		"depth": g.Depth, "rows": g.Rows, "cols": g.Cols,
		"layers_per_cell": g.LayersPerCell, "stack": g.StackLayerCount(),
		"backend": g.Exec.Backend.String(),
	})
}

//export WelvetPlaceDense
func WelvetPlaceDense(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "dense", cfgJSON)
}

func tensorFrom(inPtr *C.float, inLen C.int, shapeJSON *C.char) (*core.Tensor[float32], error) {
	n := int(inLen)
	shape := []int{1, n}
	if shapeJSON != nil {
		raw := C.GoString(shapeJSON)
		if raw != "" {
			var s []int
			if err := json.Unmarshal([]byte(raw), &s); err != nil {
				return nil, err
			}
			if len(s) > 0 {
				prod := 1
				for _, d := range s {
					prod *= d
				}
				if prod == n {
					shape = s
				}
			}
		}
	}
	x := core.NewTensor[float32](shape...)
	copy(x.Data, readF32(inPtr, n))
	return x, nil
}

//export WelvetForward
func WelvetForward(handle C.longlong, inPtr *C.float, inLen C.int, outPtr *C.float, outCap C.int) *C.char {
	return WelvetForwardEx(handle, inPtr, inLen, nil, outPtr, outCap)
}

//export WelvetForwardEx
func WelvetForwardEx(handle C.longlong, inPtr *C.float, inLen C.int, shapeJSON *C.char, outPtr *C.float, outCap C.int) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	x, err := tensorFrom(inPtr, inLen, shapeJSON)
	if err != nil {
		return errJSON(err.Error())
	}
	fwd, err := forward.Forward(g, x)
	if err != nil {
		return errJSON(err.Error())
	}
	n := len(fwd.Output.Data)
	if int(outCap) < n {
		return errJSON(fmt.Sprintf("outCap %d < %d", outCap, n))
	}
	writeF32(outPtr, fwd.Output.Data)
	return jsonOut(map[string]any{"status": "ok", "len": n, "shape": fwd.Output.Shape})
}

//export WelvetTrainSGD
func WelvetTrainSGD(handle C.longlong, inPtr *C.float, inLen C.int, tgtPtr *C.float, tgtLen C.int, lr C.double) *C.char {
	return WelvetTrainSGDEx(handle, inPtr, inLen, nil, tgtPtr, tgtLen, lr)
}

//export WelvetTrainSGDEx
func WelvetTrainSGDEx(handle C.longlong, inPtr *C.float, inLen C.int, shapeJSON *C.char, tgtPtr *C.float, tgtLen C.int, lr C.double) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	x, err := tensorFrom(inPtr, inLen, shapeJSON)
	if err != nil {
		return errJSON(err.Error())
	}
	y := core.NewTensor[float32](1, int(tgtLen))
	copy(y.Data, readF32(tgtPtr, int(tgtLen)))
	fwd, err := forward.Forward(g, x)
	if err != nil {
		return errJSON(err.Error())
	}
	loss, err := training.Step(fwd, y, float64(lr))
	if err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{"status": "ok", "loss": loss})
}

//export WelvetTrainTween
func WelvetTrainTween(handle C.longlong, inPtr *C.float, inLen C.int, shapeJSON *C.char, tgtPtr *C.float, tgtLen C.int, lr C.double) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	x, err := tensorFrom(inPtr, inLen, shapeJSON)
	if err != nil {
		return errJSON(err.Error())
	}
	y := core.NewTensor[float32](1, int(tgtLen))
	copy(y.Data, readF32(tgtPtr, int(tgtLen)))
	loss, _, err := training.StepTween(g, x, y, float64(lr))
	if err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{"status": "ok", "loss": loss})
}

//export WelvetTrainMesh
func WelvetTrainMesh(handle C.longlong, inPtr *C.float, inLen C.int, shapeJSON *C.char, tgtPtr *C.float, tgtLen C.int, ticks C.int, lr C.double) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	x, err := tensorFrom(inPtr, inLen, shapeJSON)
	if err != nil {
		return errJSON(err.Error())
	}
	y := core.NewTensor[float32](1, int(tgtLen))
	copy(y.Data, readF32(tgtPtr, int(tgtLen)))
	t := int(ticks)
	if t < 1 {
		t = 1
	}
	loss, _, err := training.StepMesh(g, x, y, t, float64(lr))
	if err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{"status": "ok", "loss": loss})
}

//export WelvetConvertDense
func WelvetConvertDense(handle C.longlong, dtype C.int, format C.int, z, y, x, l C.int) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	cell := g.At(int(z), int(y), int(x), int(l))
	if cell == nil || cell.Op == nil {
		return errJSON("no cell op")
	}
	dl, ok := cell.Op.(*dense.Layer)
	if !ok || dl == nil || dl.Weights == nil {
		return errJSON("not a dense cell")
	}
	if err := weights.Convert(dl.Weights, weights.ConvertOpts{
		DType:  core.DType(dtype),
		Format: quant.Format(format),
	}); err != nil {
		return errJSON(err.Error())
	}
	return okJSON()
}

//export WelvetSerializeEntity
func WelvetSerializeEntity(handle C.longlong, outPtr *C.uchar, outCap C.int, outLen *C.int) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	b, err := serialization.SerializeEntity(g)
	if err != nil {
		return errJSON(err.Error())
	}
	if int(outCap) < len(b) {
		return errJSON(fmt.Sprintf("outCap %d < %d", outCap, len(b)))
	}
	dst := unsafeSliceU8(outPtr, len(b))
	copy(dst, b)
	if outLen != nil {
		*outLen = C.int(len(b))
	}
	return okJSON()
}

//export WelvetDeserializeEntity
func WelvetDeserializeEntity(inPtr *C.uchar, inLen C.int) C.longlong {
	src := unsafeSliceU8(inPtr, int(inLen))
	buf := make([]byte, len(src))
	copy(buf, src)
	g, err := serialization.DeserializeEntity(buf)
	if err != nil {
		return 0
	}
	return C.longlong(storeGrid(g))
}

//export WelvetExtractDNA
func WelvetExtractDNA(handle C.longlong) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	dnaVal := dna.ExtractDNA(g)
	return jsonOut(dnaVal)
}

//export WelvetEntityByteLen
func WelvetEntityByteLen(handle C.longlong) C.int {
	g, ok := getGrid(int64(handle))
	if !ok {
		return -1
	}
	b, err := serialization.SerializeEntity(g)
	if err != nil {
		return -1
	}
	return C.int(len(b))
}
