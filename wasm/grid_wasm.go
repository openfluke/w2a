//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/tanhi"
	"github.com/openfluke/welvet/weights"
)

type gridSpec struct {
	Depth         int    `json:"depth"`
	Rows          int    `json:"rows"`
	Cols          int    `json:"cols"`
	LayersPerCell int    `json:"layers_per_cell"`
	Backend       string `json:"backend,omitempty"`
}

func registerGridGlobals() {
	js.Global().Set("createWelvetGrid", js.FuncOf(jsCreateWelvetGrid))
	js.Global().Set("createLoomNetwork", js.FuncOf(jsCreateWelvetGrid)) // Loom-compat
}

func jsCreateWelvetGrid(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return errObj("createWelvetGrid(json)")
	}
	var spec gridSpec
	raw := args[0].String()
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		// allow numeric shorthand: treat as depth=rows=cols=1 layers=N if bare number fails
		return errObj("invalid grid JSON: " + err.Error())
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
	return createGridWrapper(g)
}

func createGridWrapper(g *architecture.Grid) js.Value {
	id := storeGrid(g)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))
	obj.Set("depth", g.Depth)
	obj.Set("rows", g.Rows)
	obj.Set("cols", g.Cols)
	obj.Set("layersPerCell", g.LayersPerCell)
	obj.Set("stackLayerCount", g.StackLayerCount())

	obj.Set("getInfo", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr(map[string]any{
			"depth": g.Depth, "rows": g.Rows, "cols": g.Cols,
			"layers_per_cell": g.LayersPerCell, "stack": g.StackLayerCount(),
			"backend": g.Exec.Backend.String(),
		})
	}))

	obj.Set("setRemoteLink", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 8 {
			return errObj("setRemoteLink(z,y,x,l, tz,ty,tx,tl)")
		}
		err := g.SetRemoteLink(
			args[0].Int(), args[1].Int(), args[2].Int(), args[3].Int(),
			args[4].Int(), args[5].Int(), args[6].Int(), args[7].Int(),
		)
		if err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	obj.Set("clearRemoteLink", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 4 {
			return errObj("clearRemoteLink(z,y,x,l)")
		}
		g.ClearRemoteLink(args[0].Int(), args[1].Int(), args[2].Int(), args[3].Int())
		return okObj()
	}))
	obj.Set("convertDense", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// convertDense(dtype, format, z?,y?,x?,l?)
		if len(args) < 2 {
			return errObj("convertDense(dtype, format, z?,y?,x?,l?)")
		}
		z, y, x, l := 0, 0, 0, 0
		if len(args) >= 6 {
			z, y, x, l = args[2].Int(), args[3].Int(), args[4].Int(), args[5].Int()
		}
		s, err := denseStoreAt(g, z, y, x, l)
		if err != nil {
			return errObj(err.Error())
		}
		if err := weights.Convert(s, weights.ConvertOpts{
			DType:  core.DType(args[0].Int()),
			Format: quant.Format(args[1].Int()),
		}); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	obj.Set("applySGDDense", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// applySGDDense(Float64Array, lr?, z?,y?,x?,l?)
		if len(args) < 1 {
			return errObj("applySGDDense(dW, lr?, z?,y?,x?,l?)")
		}
		dW := readFloat64Array(args[0])
		lr := 0.1
		z, y, x, l := 0, 0, 0, 0
		idx := 1
		if len(args) > idx && args[idx].Type() == js.TypeNumber {
			lr = args[idx].Float()
			idx++
		}
		if len(args) >= idx+4 {
			z, y, x, l = args[idx].Int(), args[idx+1].Int(), args[idx+2].Int(), args[idx+3].Int()
		}
		s, err := denseStoreAt(g, z, y, x, l)
		if err != nil {
			return errObj(err.Error())
		}
		if err := s.ApplySGD(dW, lr); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))

	obj.Set("configureTanhi", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		cfg := &tanhi.UDPConfig{Enabled: true, Host: "127.0.0.1", Port: tanhi.DefaultUDPPort}
		if len(args) >= 1 && args[0].Type() == js.TypeString {
			_ = json.Unmarshal([]byte(args[0].String()), cfg)
		}
		tanhi.ConfigureGrid(g, cfg)
		return okObj()
	}))

	obj.Set("placeDense", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeDenseOnGrid(g, args)
	}))
	obj.Set("placeMHA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "mha", args)
	}))
	obj.Set("placeSwiGLU", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "swiglu", args)
	}))
	obj.Set("placeRMSNorm", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "rmsnorm", args)
	}))
	obj.Set("placeLayerNorm", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "layernorm", args)
	}))
	obj.Set("placeEmbedding", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "embedding", args)
	}))
	obj.Set("placeSoftmax", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "softmax", args)
	}))
	obj.Set("placeSequential", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "sequential", args)
	}))
	obj.Set("placeResidual", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "residual", args)
	}))
	obj.Set("placeCNN1", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "cnn1", args)
	}))
	obj.Set("placeCNN2", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "cnn2", args)
	}))
	obj.Set("placeCNN3", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "cnn3", args)
	}))
	obj.Set("placeRNN", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "rnn", args)
	}))
	obj.Set("placeLSTM", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "lstm", args)
	}))
	obj.Set("placeConvT1", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "convt1", args)
	}))
	obj.Set("placeConvT2", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "convt2", args)
	}))
	obj.Set("placeConvT3", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "convt3", args)
	}))
	obj.Set("placeParallel", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "parallel", args)
	}))
	obj.Set("placeStack", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "stack", args)
	}))
	obj.Set("placeKMeans", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "kmeans", args)
	}))
	obj.Set("placeMamba", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "mamba", args)
	}))
	obj.Set("placeMetacognition", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "metacognition", args)
	}))
	obj.Set("placeGDN", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return placeTypedOnGrid(g, "gdn", args)
	}))

	obj.Set("setMultiCore", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		mc := len(args) > 0 && args[0].Truthy()
		g.Exec.MultiCore = mc
		// Stamp every placed dense-like Op that exposes Exec via type assert.
		for z := 0; z < g.Depth; z++ {
			for y := 0; y < g.Rows; y++ {
				for x := 0; x < g.Cols; x++ {
					for l := 0; l < g.LayersPerCell; l++ {
						cell := g.At(z, y, x, l)
						if cell == nil || cell.Op == nil {
							continue
						}
						stampMultiCore(cell.Op, mc)
					}
				}
			}
		}
		return okObj()
	}))

	obj.Set("forward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridForward(g, args)
	}))
	obj.Set("backward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridBackward(g, args)
	}))
	obj.Set("trainSGD", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridTrainSGD(g, args)
	}))
	obj.Set("trainTween", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridTrainTween(g, args)
	}))
	obj.Set("trainMesh", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridTrainMesh(g, args)
	}))
	obj.Set("extractDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridExtractDNA(g)
	}))
	obj.Set("extractBlueprint", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridExtractBlueprint(g)
	}))
	obj.Set("serializeEntity", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return gridSerializeEntity(g)
	}))
	obj.Set("getDenseWeights", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		z, y, x, l := 0, 0, 0, 0
		if len(args) >= 4 {
			z, y, x, l = args[0].Int(), args[1].Int(), args[2].Int(), args[3].Int()
		}
		cell := g.At(z, y, x, l)
		if cell == nil || cell.Op == nil {
			return errObj("empty cell")
		}
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return errObj("not dense")
		}
		w, ok := dl.Weights.MasterF32()
		if !ok || w == nil {
			return errObj("no MasterF32")
		}
		return jsFloat32Array(append([]float32(nil), w...))
	}))
	obj.Set("setDenseWeights", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setDenseWeights(Float32Array, z?,y?,x?,l?)")
		}
		data := readFloat32Array(args[0])
		z, y, x, l := 0, 0, 0, 0
		if len(args) >= 5 {
			z, y, x, l = args[1].Int(), args[2].Int(), args[3].Int(), args[4].Int()
		}
		cell := g.At(z, y, x, l)
		if cell == nil || cell.Op == nil {
			return errObj("empty cell")
		}
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return errObj("not dense")
		}
		w, ok := dl.Weights.MasterF32()
		if !ok || w == nil {
			return errObj("no MasterF32")
		}
		if len(data) != len(w) {
			return errObj("weight length mismatch")
		}
		copy(w, data)
		return okObj()
	}))

	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		freeGrid(id)
		return okObj()
	}))

	return obj
}

func stampMultiCore(op any, mc bool) {
	switch v := op.(type) {
	case *dense.Layer:
		v.Exec.MultiCore = mc
		v.Core.MultiCore = mc
	case *swiglu.Layer:
		v.Exec.MultiCore = mc
	}
}
