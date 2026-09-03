//go:build js && wasm

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"syscall/js"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/stub/ensemble"
	"github.com/openfluke/welvet/stub/evaluation"
	"github.com/openfluke/welvet/stub/fountain"
	"github.com/openfluke/welvet/stub/grafting"
	"github.com/openfluke/welvet/stub/introspection"
	"github.com/openfluke/welvet/stub/memory"
	"github.com/openfluke/welvet/stub/seed"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/stub/templates"
	"github.com/openfluke/welvet/stub/universal"
	"github.com/openfluke/welvet/weights"
)

var (
	stores      = make(map[int64]*weights.Store)
	storeNextID int64 = 1
)

func storeStore(s *weights.Store) int64 {
	mu.Lock()
	id := storeNextID
	storeNextID++
	stores[id] = s
	mu.Unlock()
	return id
}

func getStore(id int64) (*weights.Store, bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := stores[id]
	return s, ok
}

func registerStubGlobals() {
	registerSeedGlobals()
	registerFountainGlobals()
	registerMemoryGlobals()
	registerHelpersGlobals()
	registerWeightsGlobals()
	registerSerializationExtras()
}

func registerSeedGlobals() {
	js.Global().Set("SeedFrom", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		parts, err := parseSeedParts(args)
		if err != nil {
			return errObj(err.Error())
		}
		return fmt.Sprintf("%d", seed.From(parts...))
	}))
	js.Global().Set("InitGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("InitGrid(gridId, seedUint64StringOrNumber)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		sd, err := readUint64Arg(args[1])
		if err != nil {
			return errObj(err.Error())
		}
		if err := seed.InitGrid(g, sd); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	js.Global().Set("GridFingerprint", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("GridFingerprint(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		return fmt.Sprintf("%d", seed.GridFingerprint(g))
	}))
	js.Global().Set("BuildDenseManifest", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("BuildDenseManifest(topologySeed, sizesJSON, dtypesJSON?)")
		}
		sd, err := readUint64Arg(args[0])
		if err != nil {
			return errObj(err.Error())
		}
		var sizes []int
		if err := json.Unmarshal([]byte(args[1].String()), &sizes); err != nil {
			return errObj("sizes: " + err.Error())
		}
		var dtypes []string
		if len(args) >= 3 && args[2].Type() == js.TypeString && args[2].String() != "" {
			_ = json.Unmarshal([]byte(args[2].String()), &dtypes)
		}
		m, err := seed.BuildDense(sd, sizes, dtypes)
		if err != nil {
			return errObj(err.Error())
		}
		b, err := seed.MarshalDense(m)
		if err != nil {
			return errObj(err.Error())
		}
		return string(b)
	}))
	js.Global().Set("BuildDenseGridFromManifest", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("BuildDenseGridFromManifest(manifestJSON)")
		}
		m, err := seed.ParseDense([]byte(args[0].String()))
		if err != nil {
			return errObj(err.Error())
		}
		g, err := seed.BuildDenseGrid(m)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(g)
	}))
	js.Global().Set("InitStoreHe", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("InitStoreHe(storeId, inputSize, seed)")
		}
		s, ok := getStore(int64(args[0].Int()))
		if !ok {
			return errObj("invalid store")
		}
		sd, err := readUint64Arg(args[2])
		if err != nil {
			return errObj(err.Error())
		}
		if err := seed.InitStoreHe(s, args[1].Int(), sd); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
}

func registerFountainGlobals() {
	js.Global().Set("FountainRecoverWeightBlobs", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("FountainRecoverWeightBlobs(blobs[], seed?, loss?, maxOverhead?)")
		}
		blobs, err := readBlobArray(args[0])
		if err != nil {
			return errObj(err.Error())
		}
		var spraySeed uint64 = 42
		loss, maxOverhead := 0.0, 3.0
		if len(args) >= 2 {
			spraySeed, _ = readUint64Arg(args[1])
		}
		if len(args) >= 3 && args[2].Type() == js.TypeNumber {
			loss = args[2].Float()
		}
		if len(args) >= 4 && args[3].Type() == js.TypeNumber {
			maxOverhead = args[3].Float()
		}
		recovered, received, sprayed, err := fountain.RecoverWeightBlobs(blobs, spraySeed, loss, maxOverhead)
		if err != nil {
			return errObj(err.Error())
		}
		arr := js.Global().Get("Array").New(len(recovered))
		for i, b := range recovered {
			arr.SetIndex(i, jsUint8Array(b))
		}
		obj := js.Global().Get("Object").New()
		obj.Set("recovered", arr)
		obj.Set("received", received)
		obj.Set("sprayed", sprayed)
		return obj
	}))
	js.Global().Set("PackGridWeights", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("PackGridWeights(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		b, err := fountain.PackGridWeights(g)
		if err != nil {
			return errObj(err.Error())
		}
		return jsUint8Array(b)
	}))
	js.Global().Set("UnpackGridWeights", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("UnpackGridWeights(gridId, Uint8Array)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		if err := fountain.UnpackGridWeights(g, readUint8Array(args[1])); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	js.Global().Set("FountainLTRoundTrip", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// FountainLTRoundTrip(sources[]Uint8Array, seed?) — peel with 0 loss
		if len(args) < 1 {
			return errObj("FountainLTRoundTrip(sources[], seed?)")
		}
		srcs, err := readBlobArray(args[0])
		if err != nil {
			return errObj(err.Error())
		}
		var spraySeed uint64 = 7
		if len(args) >= 2 {
			spraySeed, _ = readUint64Arg(args[1])
		}
		recovered, received, sprayed, err := fountain.RecoverWeightBlobs(srcs, spraySeed, 0, 2.5)
		if err != nil {
			return errObj(err.Error())
		}
		ok := fountain.BlocksEqual(srcs, recovered)
		obj := js.Global().Get("Object").New()
		obj.Set("ok", ok)
		obj.Set("received", received)
		obj.Set("sprayed", sprayed)
		obj.Set("k", len(srcs))
		return obj
	}))
}

func registerMemoryGlobals() {
	js.Global().Set("MemoryFromGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("MemoryFromGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		fp := memory.FromGrid(g)
		return jsonStr(fp)
	}))
	js.Global().Set("ReleaseTransient", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		memory.ReleaseTransient()
		return okObj()
	}))
	js.Global().Set("SetMemoryHistoryRecording", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		on := len(args) > 0 && args[0].Truthy()
		memory.SetMemoryHistoryRecording(on)
		return okObj()
	}))
}

func registerHelpersGlobals() {
	js.Global().Set("GraftGrids", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("GraftGrids(gridIds[], combine?)")
		}
		gs, err := readGridSlice(args[0])
		if err != nil {
			return errObj(err.Error())
		}
		combine := parallel.CombineConcat
		if len(args) >= 2 && args[1].Type() == js.TypeString {
			combine = parallel.CombineMode(args[1].String())
		}
		layer, err := grafting.GraftGrids(gs, combine)
		if err != nil {
			return errObj(err.Error())
		}
		return createParallelWrapper(layer)
	}))
	js.Global().Set("GraftToGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("GraftToGrid(gridIds[])")
		}
		gs, err := readGridSlice(args[0])
		if err != nil {
			return errObj(err.Error())
		}
		out, err := grafting.GraftToGrid(gs)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(out)
	}))
	js.Global().Set("ResidualGraftGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ResidualGraftGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		out, err := grafting.ResidualGraftGrid(g)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(out)
	}))
	js.Global().Set("TemplateBuildPrompt", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		name := "chatml"
		user := ""
		system := ""
		if len(args) >= 1 {
			name = args[0].String()
		}
		if len(args) >= 2 {
			user = args[1].String()
		}
		if len(args) >= 3 {
			system = args[2].String()
		}
		t := templateByName(name)
		return t.BuildPrompt(nil, system, user)
	}))
	js.Global().Set("listWelvetTemplates", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr([]string{"chatml", "plain", "bitnet-inst", "microsoft-bitnet", "llama3"})
	}))
	js.Global().Set("EnsembleMajorityVote", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("EnsembleMajorityVote([[class,...],...])")
		}
		var votes [][]int
		if err := json.Unmarshal([]byte(args[0].String()), &votes); err != nil {
			return errObj(err.Error())
		}
		return jsonStr(ensemble.MajorityVote(votes))
	}))
	js.Global().Set("EvaluatePrediction", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("EvaluatePrediction(sampleIndex, expected, actual)")
		}
		r := evaluation.EvaluatePrediction(args[0].Int(), args[1].Float(), args[2].Float())
		return jsonStr(r)
	}))
	js.Global().Set("IntrospectGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("IntrospectGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		s, err := introspection.GetMethodsJSON(g)
		if err != nil {
			return errObj(err.Error())
		}
		return s
	}))
	js.Global().Set("ProbeDeepGeometry", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ProbeDeepGeometry(geomsJSON)")
		}
		var geoms []universal.TensorMeta
		if err := json.Unmarshal([]byte(args[0].String()), &geoms); err != nil {
			return errObj(err.Error())
		}
		arch, order := universal.ProbeDeepGeometry(geoms)
		return jsonStr(map[string]any{"archetypes": arch, "order": order})
	}))
}

func registerWeightsGlobals() {
	js.Global().Set("createWelvetStore", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// createWelvetStore(rows, cols, dtype, format, Float32Array?)
		if len(args) < 2 {
			return errObj("createWelvetStore(rows, cols, dtype?, format?, dataF32?)")
		}
		rows, cols := args[0].Int(), args[1].Int()
		dt := core.DTypeFloat32
		format := quant.FormatNone
		if len(args) >= 3 && args[2].Type() == js.TypeNumber {
			dt = core.DType(args[2].Int())
		}
		if len(args) >= 4 && args[3].Type() == js.TypeNumber {
			format = quant.Format(args[3].Int())
		}
		n := rows * cols
		data := make([]float32, n)
		if len(args) >= 5 && args[4].Type() == js.TypeObject {
			src := readFloat32Array(args[4])
			copy(data, src)
		} else {
			for i := range data {
				data[i] = float32(math.Sin(float64(i)*0.17)) * 0.5
			}
		}
		s, err := weights.New(rows, cols, data, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return errObj(err.Error())
		}
		if dt != core.DTypeFloat32 {
			if err := s.SetDType(dt); err != nil {
				return errObj(err.Error())
			}
		}
		if format != quant.FormatNone {
			if err := s.Pack(format); err != nil {
				return errObj(err.Error())
			}
		}
		return createStoreWrapper(s)
	}))
	js.Global().Set("listWelvetConcreteTrainModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes := parallel.AllConcreteTrainModes()
		out := make([]string, len(modes))
		for i, m := range modes {
			out[i] = m.String()
		}
		return jsonStr(out)
	}))
	js.Global().Set("listWelvetCreditTrainModes", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modes := parallel.AllCreditTrainModes()
		out := make([]string, len(modes))
		for i, m := range modes {
			out[i] = m.String()
		}
		return jsonStr(out)
	}))
	js.Global().Set("ParseTrainMode", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ParseTrainMode(name)")
		}
		m, err := parallel.ParseTrainMode(args[0].String())
		if err != nil {
			return errObj(err.Error())
		}
		return m.String()
	}))
}

func registerSerializationExtras() {
	js.Global().Set("DeserializeEntity", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("DeserializeEntity(Uint8Array)")
		}
		g, err := serialization.DeserializeEntity(readUint8Array(args[0]))
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(g)
	}))
	js.Global().Set("SerializeEntity", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("SerializeEntity(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		b, err := serialization.SerializeEntity(g)
		if err != nil {
			return errObj(err.Error())
		}
		return jsUint8Array(b)
	}))
	js.Global().Set("BuildSpec", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("BuildSpec(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		spec, err := serialization.BuildSpec(g)
		if err != nil {
			return errObj(err.Error())
		}
		return jsonStr(spec)
	}))
	js.Global().Set("PackableFormats", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		fmts := serialization.PackableFormats()
		out := make([]map[string]any, len(fmts))
		for i, f := range fmts {
			out[i] = map[string]any{"id": int(f), "name": f.String()}
		}
		return jsonStr(out)
	}))
}

func createStoreWrapper(s *weights.Store) js.Value {
	id := storeStore(s)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))
	obj.Set("rows", s.Rows)
	obj.Set("cols", s.Cols)
	obj.Set("dtype", int(s.DType))
	obj.Set("format", int(s.Format))

	obj.Set("applySGD", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("applySGD(Float64Array|number[], lr?)")
		}
		dW := readFloat64Array(args[0])
		lr := 0.1
		if len(args) >= 2 && args[1].Type() == js.TypeNumber {
			lr = args[1].Float()
		}
		if err := s.ApplySGD(dW, lr); err != nil {
			return errObj(err.Error())
		}
		return okObj()
	}))
	obj.Set("setDType", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("setDType(dtype)")
		}
		if err := s.SetDType(core.DType(args[0].Int())); err != nil {
			return errObj(err.Error())
		}
		obj.Set("dtype", int(s.DType))
		return okObj()
	}))
	obj.Set("pack", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("pack(format)")
		}
		if err := s.Pack(quant.Format(args[0].Int())); err != nil {
			return errObj(err.Error())
		}
		obj.Set("format", int(s.Format))
		return okObj()
	}))
	obj.Set("convert", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("convert(dtype, format)")
		}
		if err := weights.Convert(s, weights.ConvertOpts{
			DType:  core.DType(args[0].Int()),
			Format: quant.Format(args[1].Int()),
		}); err != nil {
			return errObj(err.Error())
		}
		obj.Set("dtype", int(s.DType))
		obj.Set("format", int(s.Format))
		return okObj()
	}))
	obj.Set("flattenF32", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		f, err := s.FlattenF32()
		if err != nil {
			return errObj(err.Error())
		}
		return jsFloat32Array(f)
	}))
	obj.Set("retainsF32Master", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return s.RetainsF32Master()
	}))
	obj.Set("f32BufferLen", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return s.F32BufferLen()
	}))
	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		mu.Lock()
		delete(stores, id)
		mu.Unlock()
		return okObj()
	}))
	return obj
}

func templateByName(name string) templates.Template {
	switch name {
	case "llama3":
		return templates.Llama3
	case "plain":
		return templates.PlainCompletion
	case "bitnet-inst":
		return templates.BitNetInstruction
	case "microsoft-bitnet":
		return templates.MicrosoftBitNetChat
	default:
		return templates.ChatML
	}
}

func parseSeedParts(args []js.Value) ([]any, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("SeedFrom needs parts")
	}
	if args[0].Type() == js.TypeString {
		var raw []any
		if err := json.Unmarshal([]byte(args[0].String()), &raw); err != nil {
			// single string seed part
			return []any{args[0].String()}, nil
		}
		out := make([]any, 0, len(raw))
		for _, v := range raw {
			switch t := v.(type) {
			case float64:
				if t == math.Trunc(t) && t >= math.MinInt64 && t <= math.MaxInt64 {
					out = append(out, int64(t))
				} else {
					out = append(out, t)
				}
			case bool, string:
				out = append(out, t)
			default:
				out = append(out, fmt.Sprint(t))
			}
		}
		return out, nil
	}
	out := make([]any, 0, len(args))
	for _, a := range args {
		switch a.Type() {
		case js.TypeString:
			out = append(out, a.String())
		case js.TypeBoolean:
			out = append(out, a.Truthy())
		case js.TypeNumber:
			f := a.Float()
			if f == math.Trunc(f) {
				out = append(out, int64(f))
			} else {
				out = append(out, f)
			}
		default:
			out = append(out, a.String())
		}
	}
	return out, nil
}

func readUint64Arg(v js.Value) (uint64, error) {
	switch v.Type() {
	case js.TypeString:
		return strconv.ParseUint(v.String(), 10, 64)
	case js.TypeNumber:
		f := v.Float()
		if f < 0 || f > float64(^uint64(0)>>11) {
			// still accept as uint64 truncation of integer part
		}
		return uint64(f), nil
	default:
		return 0, fmt.Errorf("uint64 arg must be string or number")
	}
}

func readFloat64Array(jsVal js.Value) []float64 {
	if jsVal.IsUndefined() || jsVal.IsNull() {
		return nil
	}
	length := jsVal.Get("length").Int()
	out := make([]float64, length)
	for i := 0; i < length; i++ {
		out[i] = jsVal.Index(i).Float()
	}
	return out
}

func readBlobArray(jsVal js.Value) ([][]byte, error) {
	if jsVal.Type() != js.TypeObject {
		return nil, fmt.Errorf("expected array of Uint8Array")
	}
	n := jsVal.Get("length").Int()
	out := make([][]byte, n)
	for i := 0; i < n; i++ {
		out[i] = readUint8Array(jsVal.Index(i))
	}
	return out, nil
}

func readGridSlice(jsVal js.Value) ([]*architecture.Grid, error) {
	if jsVal.Type() == js.TypeString {
		var ids []int64
		if err := json.Unmarshal([]byte(jsVal.String()), &ids); err != nil {
			return nil, err
		}
		out := make([]*architecture.Grid, 0, len(ids))
		for _, id := range ids {
			g, ok := getGrid(id)
			if !ok {
				return nil, fmt.Errorf("invalid grid id %d", id)
			}
			out = append(out, g)
		}
		return out, nil
	}
	n := jsVal.Get("length").Int()
	out := make([]*architecture.Grid, 0, n)
	for i := 0; i < n; i++ {
		id := int64(jsVal.Index(i).Int())
		g, ok := getGrid(id)
		if !ok {
			return nil, fmt.Errorf("invalid grid id %d", id)
		}
		out = append(out, g)
	}
	return out, nil
}

// denseStoreAt returns the Dense Weights store at a cell (for convert/ApplySGD).
func denseStoreAt(g *architecture.Grid, z, y, x, l int) (*weights.Store, error) {
	cell := g.At(z, y, x, l)
	if cell == nil || cell.Op == nil {
		return nil, fmt.Errorf("empty cell")
	}
	dl, ok := cell.Op.(*dense.Layer)
	if !ok {
		return nil, fmt.Errorf("not dense")
	}
	if dl.Weights == nil {
		return nil, fmt.Errorf("nil weights")
	}
	return dl.Weights, nil
}
