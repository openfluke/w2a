//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

func registerParallelGlobals() {
	js.Global().Set("listWelvetTrainModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes := parallel.AllTrainModes()
		out := make([]string, len(modes))
		for i, m := range modes {
			out[i] = m.String()
		}
		return jsonStr(out)
	}))
	js.Global().Set("listWelvetNamedTrainModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes := parallel.AllNamedTrainModes()
		out := make([]string, len(modes))
		for i, m := range modes {
			out[i] = m.String()
		}
		return jsonStr(out)
	}))
	js.Global().Set("createWelvetBicameral", js.FuncOf(jsCreateBicameral))
	js.Global().Set("createWelvetHemispheres", js.FuncOf(jsCreateHemispheres))
	js.Global().Set("createWelvetParallel", js.FuncOf(jsCreateParallel))
}

func jsCreateBicameral(this js.Value, args []js.Value) interface{} {
	in, hidden, out := 8, 8, 8
	act := core.ActivationReLU
	dt := core.DTypeFloat32
	format := quant.FormatNone
	if len(args) >= 1 && args[0].Type() == js.TypeString {
		var s struct {
			In     int    `json:"in"`
			Hidden int    `json:"hidden"`
			Out    int    `json:"out"`
			Act    string `json:"act"`
			DType  int    `json:"dtype"`
			Format int    `json:"format"`
		}
		_ = json.Unmarshal([]byte(args[0].String()), &s)
		if s.In > 0 {
			in = s.In
		}
		if s.Hidden > 0 {
			hidden = s.Hidden
		}
		if s.Out > 0 {
			out = s.Out
		}
		if s.Act != "" {
			act = core.ParseActivation(s.Act)
		}
		if s.DType != 0 {
			dt = core.DType(s.DType)
		}
		format = quant.Format(s.Format)
	} else {
		if len(args) >= 1 {
			in = args[0].Int()
		}
		if len(args) >= 2 {
			hidden = args[1].Int()
		}
		if len(args) >= 3 {
			out = args[2].Int()
		}
	}
	st, err := parallel.Bicameral(in, hidden, out, act, dt, format)
	if err != nil {
		return errObj(err.Error())
	}
	return createStackWrapper(st)
}

func jsCreateHemispheres(this js.Value, args []js.Value) interface{} {
	dim, n := 8, 2
	combine := parallel.CombineAdd
	if len(args) >= 1 && args[0].Type() == js.TypeString {
		var s struct {
			Dim     int    `json:"dim"`
			N       int    `json:"n"`
			Combine string `json:"combine"`
		}
		_ = json.Unmarshal([]byte(args[0].String()), &s)
		if s.Dim > 0 {
			dim = s.Dim
		}
		if s.N > 0 {
			n = s.N
		}
		if s.Combine != "" {
			combine = parallel.CombineMode(s.Combine)
		}
	}
	layer, err := parallel.Hemispheres(dim, dim, n, combine, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return errObj(err.Error())
	}
	return createParallelWrapper(layer)
}

func jsCreateParallel(this js.Value, args []js.Value) interface{} {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineAdd}
	if len(args) >= 1 {
		_ = json.Unmarshal([]byte(args[0].String()), &cfg)
	}
	layer, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil {
		return errObj(err.Error())
	}
	return createParallelWrapper(layer)
}

func createStackWrapper(s *parallel.Stack) js.Value {
	id := storeStack(s)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))
	obj.Set("kind", "stack")
	obj.Set("children", len(s.Children))

	obj.Set("setChildModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes, err := parseModes(args)
		if err != nil {
			return errObj(err.Error())
		}
		s.SetChildModes(modes...)
		return okObj()
	}))
	obj.Set("setTanhi", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setTanhi(cfgJSON)")
		}
		s.SetTanhi(parseTanhiCfg(args[0].String()))
		return okObj()
	}))
	obj.Set("trainStackMSE", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return trainStackMSE(s, args)
	}))
	obj.Set("trainStackCE", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return trainStackCE(s, args)
	}))
	obj.Set("forward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		in, errV := mustTensor(args, 0)
		if errV != nil {
			return errV
		}
		_, post, err := parallel.ForwardStack(s, in)
		if err != nil {
			return errObj(err.Error())
		}
		out := js.Global().Get("Object").New()
		out.Set("output", jsFloat32Array(post.Data))
		out.Set("shape", jsonStr(post.Shape))
		return out
	}))
	obj.Set("placeOnGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("placeOnGrid(gridId, z?,y?,x?,l?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		z, y, x, l := 0, 0, 0, 0
		if len(args) >= 5 {
			z, y, x, l = args[1].Int(), args[2].Int(), args[3].Int(), args[4].Int()
		}
		if err := parallel.PlaceStack(g, z, y, x, l, s); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		mu.Lock()
		delete(stacks, id)
		mu.Unlock()
		return okObj()
	}))
	return obj
}

func createParallelWrapper(l *parallel.Layer) js.Value {
	id := storeParallel(l)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))
	obj.Set("kind", "parallel")
	obj.Set("branches", len(l.Branches))

	obj.Set("setBranchModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes, err := parseModes(args)
		if err != nil {
			return errObj(err.Error())
		}
		l.SetBranchModes(modes...)
		return okObj()
	}))
	obj.Set("setTanhi", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setTanhi(cfgJSON)")
		}
		l.SetTanhi(parseTanhiCfg(args[0].String()))
		return okObj()
	}))
	obj.Set("setCamSync", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setCamSync(cfgJSON)")
		}
		var cfg parallel.CamSyncConfig
		if err := json.Unmarshal([]byte(args[0].String()), &cfg); err != nil {
			return errObj(err.Error())
		}
		l.CamSync = &cfg
		return okObj()
	}))
	obj.Set("setCamKit", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setCamKit(cfgJSON)")
		}
		var kit parallel.CamKit
		if err := json.Unmarshal([]byte(args[0].String()), &kit); err != nil {
			return errObj(err.Error())
		}
		l.SetCamKit(kit)
		return okObj()
	}))
	obj.Set("trainMSE", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return trainParallelMSE(l, args)
	}))
	obj.Set("forward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		in, errV := mustTensor(args, 0)
		if errV != nil {
			return errV
		}
		_, post, err := parallel.Forward(l, in)
		if err != nil {
			return errObj(err.Error())
		}
		out := js.Global().Get("Object").New()
		out.Set("output", jsFloat32Array(post.Data))
		return out
	}))
	obj.Set("placeOnGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("placeOnGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		if err := parallel.Place(g, 0, 0, 0, 0, l); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		mu.Lock()
		delete(parallels, id)
		mu.Unlock()
		return okObj()
	}))
	return obj
}

func parseModes(args []js.Value) ([]parallel.TrainMode, error) {
	var names []string
	if len(args) == 1 && args[0].Type() == js.TypeString {
		if err := json.Unmarshal([]byte(args[0].String()), &names); err != nil {
			// single mode name
			m, err2 := parallel.ParseTrainMode(args[0].String())
			if err2 != nil {
				return nil, err
			}
			return []parallel.TrainMode{m}, nil
		}
	} else {
		for _, a := range args {
			names = append(names, a.String())
		}
	}
	out := make([]parallel.TrainMode, len(names))
	for i, n := range names {
		m, err := parallel.ParseTrainMode(n)
		if err != nil {
			return nil, err
		}
		out[i] = m
	}
	return out, nil
}

func modeFromArgs(args []js.Value, defaultMode parallel.TrainMode) parallel.TrainMode {
	for i := len(args) - 1; i >= 0; i-- {
		if args[i].Type() == js.TypeString {
			m, err := parallel.ParseTrainMode(args[i].String())
			if err == nil {
				return m
			}
		}
	}
	return defaultMode
}

func trainStackMSE(s *parallel.Stack, args []js.Value) interface{} {
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
	mode := modeFromArgs(args, parallel.ModeNormalBP)
	lr := 0.01
	if args[len(args)-1].Type() == js.TypeNumber {
		lr = args[len(args)-1].Float()
	}
	loss, err := parallel.TrainStackMSE(s, in, target, mode, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	out.Set("mode", mode.String())
	return out
}

func trainStackCE(s *parallel.Stack, args []js.Value) interface{} {
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
	mode := modeFromArgs(args, parallel.ModeNormalBP)
	lr := 0.01
	if args[len(args)-1].Type() == js.TypeNumber {
		lr = args[len(args)-1].Float()
	}
	loss, err := parallel.TrainStackCE(s, in, target, mode, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	out.Set("mode", mode.String())
	return out
}

func trainParallelMSE(l *parallel.Layer, args []js.Value) interface{} {
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
	mode := modeFromArgs(args, parallel.ModeNormalBP)
	lr := 0.01
	if args[len(args)-1].Type() == js.TypeNumber {
		lr = args[len(args)-1].Float()
	}
	loss, err := parallel.TrainMSE(l, in, target, mode, lr)
	if err != nil {
		return errObj(err.Error())
	}
	out := js.Global().Get("Object").New()
	out.Set("loss", loss)
	out.Set("mode", mode.String())
	return out
}
