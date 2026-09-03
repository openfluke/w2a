//go:build js && wasm

package main

import (
	"syscall/js"

	"github.com/openfluke/welvet/model/sampling"
	"github.com/openfluke/welvet/model/tokenizer"
	"github.com/openfluke/welvet/stub/serialization"
)

func registerModelGlobals() {
	js.Global().Set("SerializeGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("SerializeGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		b, err := serialization.SerializeNetwork(g)
		if err != nil {
			return errObj(err.Error())
		}
		return jsUint8Array(b)
	}))
	js.Global().Set("DeserializeGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("DeserializeGrid(bytes)")
		}
		raw := readUint8Array(args[0])
		// Accept both JSON network and binary .entity (magic often starts with 'E').
		if g, err := serialization.DeserializeEntity(raw); err == nil {
			return createGridWrapper(g)
		}
		g, err := serialization.DeserializeNetwork(raw)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(g)
	}))
	js.Global().Set("ArgMax", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ArgMax(Float32Array)")
		}
		data := readFloat32Array(args[0])
		return sampling.ArgMax(data)
	}))
	js.Global().Set("SampleTopK", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("SampleTopK(logits, k, temperature?)")
		}
		data := readFloat32Array(args[0])
		k := args[1].Int()
		temp := 1.0
		if len(args) >= 3 {
			temp = args[2].Float()
		}
		idx := sampling.SampleTopK(data, k, float32(temp), false)
		return idx
	}))
	js.Global().Set("NewTokenizerFromJSON", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("NewTokenizerFromJSON(jsonBytesOrString)")
		}
		var raw []byte
		if args[0].Type() == js.TypeString {
			raw = []byte(args[0].String())
		} else {
			raw = readUint8Array(args[0])
		}
		tok, err := tokenizer.NewTokenizerFromJSON(raw)
		if err != nil {
			return errObj(err.Error())
		}
		obj := js.Global().Get("Object").New()
		obj.Set("encode", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			text := ""
			if len(a) > 0 {
				text = a[0].String()
			}
			ids := tok.Encode(text, len(a) < 2 || a[1].Truthy())
			arr := js.Global().Get("Uint32Array").New(len(ids))
			for i, id := range ids {
				arr.SetIndex(i, id)
			}
			return arr
		}))
		obj.Set("decode", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			if len(a) < 1 {
				return ""
			}
			n := a[0].Get("length").Int()
			ids := make([]uint32, n)
			for i := 0; i < n; i++ {
				ids[i] = uint32(a[0].Index(i).Int())
			}
			skip := len(a) < 2 || a[1].Truthy()
			return tok.Decode(ids, skip)
		}))
		return obj
	}))
	// Full transformer.LoadEntity pulls fusedgpu/WebGPU; expose a light prompt helper only.
	js.Global().Set("BuildTransformerPrompt", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		user := "hello"
		if len(args) >= 1 {
			user = args[0].String()
		}
		sys := ""
		if len(args) >= 2 {
			sys = args[1].String()
		}
		if sys == "" {
			return user
		}
		return sys + "\n\n" + user
	}))
}
