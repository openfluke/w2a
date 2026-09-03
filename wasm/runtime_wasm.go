//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/telemetry"
	"github.com/openfluke/welvet/systems/tween"
)

// lastFwd caches the last forward tape per grid id for backward without re-forward.
var lastFwd = map[int64]*forward.Result[float32]{}

func registerRuntimeGlobals() {
	js.Global().Set("Forward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("Forward(gridId, data, shapeJson?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		return gridForward(g, args[1:])
	}))
	js.Global().Set("TrainStep", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("TrainStep(gridId, input, target, lr?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		return gridTrainSGD(g, args[1:])
	}))
	js.Global().Set("TrainStepTween", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("TrainStepTween(gridId, input, target, lr?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		return gridTrainTween(g, args[1:])
	}))
	js.Global().Set("TrainStepMesh", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("TrainStepMesh(gridId, input, target, ticks?, lr?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		return gridTrainMesh(g, args[1:])
	}))
	js.Global().Set("createWelvetStepState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("createWelvetStepState(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		st := step.New[float32](g)
		id := storeStep(st)
		obj := js.Global().Get("Object").New()
		obj.Set("_id", float64(id))
		obj.Set("setInput", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			t, errV := mustTensor(a, 0)
			if errV != nil {
				return errV
			}
			st.SetInput(t)
			return okObj()
		}))
		obj.Set("step", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			capture := len(a) > 0 && a[0].Truthy()
			dur, err := step.Forward(g, st, capture)
			if err != nil {
				return errObj(err.Error())
			}
			out := js.Global().Get("Object").New()
			out.Set("ms", float64(dur.Microseconds())/1000.0)
			out.Set("stepCount", float64(st.StepCount))
			return out
		}))
		obj.Set("free", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			mu.Lock()
			delete(stepStates, id)
			mu.Unlock()
			return okObj()
		}))
		return obj
	}))
	js.Global().Set("createWelvetTweenState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("createWelvetTweenState(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid id")
		}
		st := tween.NewState[float32](g, tween.DefaultConfig())
		id := storeTween(st)
		obj := js.Global().Get("Object").New()
		obj.Set("_id", float64(id))
		obj.Set("free", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			mu.Lock()
			delete(tweenStates, id)
			mu.Unlock()
			return okObj()
		}))
		return obj
	}))
}

func mustTensor(args []js.Value, dataIdx int) (*core.Tensor[float32], interface{}) {
	if len(args) <= dataIdx {
		return nil, errObj("missing Float32Array data")
	}
	data := readFloat32Array(args[dataIdx])
	shape := []int{1, len(data)}
	shapeIdx := dataIdx + 1
	if len(args) > shapeIdx && !args[shapeIdx].IsUndefined() && args[shapeIdx].Type() != js.TypeNumber {
		if args[shapeIdx].Type() == js.TypeString {
			_ = json.Unmarshal([]byte(args[shapeIdx].String()), &shape)
		} else if args[shapeIdx].Type() == js.TypeObject &&
			args[shapeIdx].Get("length").Truthy() &&
			!args[shapeIdx].Get("BYTES_PER_ELEMENT").Truthy() {
			// Plain JS array of ints — not a TypedArray (those are data tensors).
			shape = readIntSlice(args[shapeIdx])
		}
	}
	n := 1
	for _, d := range shape {
		if d > 0 {
			n *= d
		}
	}
	if n != len(data) {
		shape = []int{1, len(data)}
	}
	t := core.NewTensor[float32](shape...)
	copy(t.Data, data)
	return t, nil
}

func gridForward(g *architecture.Grid, args []js.Value) interface{} {
	in, errV := mustTensor(args, 0)
	if errV != nil {
		return errV
	}
	fwd, err := forward.Forward(g, in)
	if err != nil {
		return errObj(err.Error())
	}
	// cache by scanning registries
	mu.RLock()
	for id, gg := range grids {
		if gg == g {
			lastFwd[id] = fwd
			break
		}
	}
	mu.RUnlock()
	out := js.Global().Get("Object").New()
	out.Set("output", jsFloat32Array(fwd.Output.Data))
	out.Set("shape", jsonStr(fwd.Output.Shape))
	out.Set("steps", len(fwd.Steps))
	return out
}

func gridBackward(g *architecture.Grid, args []js.Value) interface{} {
	gy, errV := mustTensor(args, 0)
	if errV != nil {
		return errV
	}
	var fwd *forward.Result[float32]
	mu.RLock()
	for id, gg := range grids {
		if gg == g {
			fwd = lastFwd[id]
			break
		}
	}
	mu.RUnlock()
	if fwd == nil {
		return errObj("backward: no cached forward — call forward first")
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	if bwd.GradIn != nil {
		out.Set("gradIn", jsFloat32Array(bwd.GradIn.Data))
	}
	out.Set("gradWs", len(bwd.GradWs))
	return out
}

func gridTrainSGD(g *architecture.Grid, args []js.Value) interface{} {
	in, errV := mustTensor(args, 0)
	if errV != nil {
		return errV
	}
	// target: first Float32Array after optional shape string/array
	targetIdx := 1
	for i := 1; i < len(args); i++ {
		if args[i].Type() == js.TypeObject && args[i].Get("BYTES_PER_ELEMENT").Truthy() {
			targetIdx = i
			break
		}
	}
	// If input used a shape string at args[1], rebuild input with that shape.
	if targetIdx > 1 && args[1].Type() == js.TypeString {
		in2, errV2 := mustTensor([]js.Value{args[0], args[1]}, 0)
		if errV2 == nil {
			in = in2
		}
	}
	target, errV := mustTensor(args, targetIdx)
	if errV != nil {
		return errV
	}
	lr := 0.01
	if len(args) > targetIdx+1 && args[targetIdx+1].Type() == js.TypeNumber {
		lr = args[targetIdx+1].Float()
	} else if len(args) > 2 && args[len(args)-1].Type() == js.TypeNumber {
		lr = args[len(args)-1].Float()
	}
	fwd, err := forward.Forward(g, in)
	if err != nil {
		return errObj(err.Error())
	}
	// Align target rank/shape to forward output when element counts match (MSE is shape-strict).
	if target != nil && fwd.Output != nil && len(target.Data) == len(fwd.Output.Data) {
		target.Shape = append([]int(nil), fwd.Output.Shape...)
	}
	loss, err := training.Step(fwd, target, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	out.Set("output", jsFloat32Array(fwd.Output.Data))
	return out
}

func gridTrainTween(g *architecture.Grid, args []js.Value) interface{} {
	in, errV := mustTensor(args, 0)
	if errV != nil {
		return errV
	}
	targetIdx := 1
	for i := 1; i < len(args); i++ {
		if args[i].Type() == js.TypeObject && args[i].Get("BYTES_PER_ELEMENT").Truthy() {
			targetIdx = i
		}
	}
	target, errV := mustTensor(args, targetIdx)
	if errV != nil {
		return errV
	}
	lr := 0.01
	if len(args) > 0 && args[len(args)-1].Type() == js.TypeNumber {
		lr = args[len(args)-1].Float()
	}
	loss, _, err := training.StepTween(g, in, target, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	return out
}

func gridTrainMesh(g *architecture.Grid, args []js.Value) interface{} {
	in, errV := mustTensor(args, 0)
	if errV != nil {
		return errV
	}
	targetIdx := 1
	for i := 1; i < len(args); i++ {
		if args[i].Type() == js.TypeObject && args[i].Get("BYTES_PER_ELEMENT").Truthy() {
			targetIdx = i
		}
	}
	target, errV := mustTensor(args, targetIdx)
	if errV != nil {
		return errV
	}
	ticks := 1
	lr := 0.01
	for i := targetIdx + 1; i < len(args); i++ {
		if args[i].Type() == js.TypeNumber {
			if ticks == 1 && args[i].Int() >= 1 && args[i].Float() == float64(args[i].Int()) && args[i].Int() < 1000 {
				ticks = args[i].Int()
			} else {
				lr = args[i].Float()
			}
		}
	}
	loss, _, err := training.StepMesh(g, in, target, ticks, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	return out
}

func gridExtractDNA(g *architecture.Grid) interface{} {
	return jsonStr(dna.ExtractDNA(g))
}

func gridExtractBlueprint(g *architecture.Grid) interface{} {
	return jsonStr(telemetry.ExtractNetworkBlueprint(g, "wasm"))
}

func gridSerializeEntity(g *architecture.Grid) interface{} {
	b, err := serialization.SerializeEntity(g)
	if err != nil {
		// Fall back to JSON network bytes if entity codec rejects the ops.
		b2, err2 := serialization.SerializeNetwork(g)
		if err2 != nil {
			return errObj(err.Error() + "; json: " + err2.Error())
		}
		return jsUint8Array(b2)
	}
	return jsUint8Array(b)
}
