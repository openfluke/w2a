//go:build js && wasm

package main

import (
	"syscall/js"

	"github.com/openfluke/welvet/lucy"
)

func registerLucyGlobals() {
	js.Global().Set("LucyAvailability", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("LucyAvailability(inferMs, trainMs)")
		}
		return lucy.Availability(args[0].Float(), args[1].Float())
	}))
	js.Global().Set("LucyScore", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return errObj("LucyScore(throughput, availability, acc)")
		}
		return lucy.Score(args[0].Float(), args[1].Float(), args[2].Float())
	}))
	js.Global().Set("LucySoftAccBatch", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("LucySoftAccBatch(pred, target)")
		}
		return lucy.SoftAccBatch(readFloat32Array(args[0]), readFloat32Array(args[1]))
	}))
}
