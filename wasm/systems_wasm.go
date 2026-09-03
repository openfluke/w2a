//go:build js && wasm

package main

import (
	"encoding/json"
	"syscall/js"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/evolution"
	"github.com/openfluke/welvet/systems/tanhi"
	"github.com/openfluke/welvet/systems/telemetry"
)

func parseTanhiCfg(raw string) *tanhi.UDPConfig {
	cfg := &tanhi.UDPConfig{Enabled: true, Host: "127.0.0.1", Port: tanhi.DefaultUDPPort}
	_ = json.Unmarshal([]byte(raw), cfg)
	return cfg
}

func registerSystemsGlobals() {
	js.Global().Set("ConfigureTanhi", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ConfigureTanhi(gridId, cfgJSON?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		cfg := &tanhi.UDPConfig{Enabled: true, Host: "127.0.0.1", Port: tanhi.DefaultUDPPort}
		if len(args) >= 2 {
			_ = json.Unmarshal([]byte(args[1].String()), cfg)
		}
		tanhi.ConfigureGrid(g, cfg)
		return okObj()
	}))
	js.Global().Set("EmitSweep", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		cfg := &tanhi.UDPConfig{Enabled: false}
		label := "sweep"
		if len(args) >= 1 {
			label = args[0].String()
		}
		if len(args) >= 2 {
			_ = json.Unmarshal([]byte(args[1].String()), cfg)
		}
		tanhi.EmitSweep(cfg, label)
		return okObj()
	}))
	js.Global().Set("ExtractDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ExtractDNA(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		return jsonStr(dna.ExtractDNA(g))
	}))
	js.Global().Set("CompareDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("CompareDNA(dnaJSON_A, dnaJSON_B)")
		}
		var a, b dna.NetworkDNA
		if err := json.Unmarshal([]byte(args[0].String()), &a); err != nil {
			return errObj(err.Error())
		}
		if err := json.Unmarshal([]byte(args[1].String()), &b); err != nil {
			return errObj(err.Error())
		}
		return jsonStr(dna.CompareNetworks(a, b))
	}))
	js.Global().Set("createWelvetNEATPopulation", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("createWelvetNEATPopulation(gridId, size, cfgJSON?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		size := args[1].Int()
		cfg := evolution.DefaultNEATConfig(64)
		if len(args) >= 3 {
			_ = json.Unmarshal([]byte(args[2].String()), &cfg)
		}
		pop, err := evolution.NewNEATPopulation(g, size, cfg)
		if err != nil {
			return errObj(err.Error())
		}
		obj := js.Global().Get("Object").New()
		obj.Set("size", len(pop.Networks))
		obj.Set("evolveWithFitnesses", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			if len(a) < 1 {
				return errObj("evolveWithFitnesses(Float64Array)")
			}
			n := a[0].Get("length").Int()
			fits := make([]float64, n)
			for i := 0; i < n; i++ {
				fits[i] = a[0].Index(i).Float()
			}
			i := 0
			err := pop.Evolve(func(_ *architecture.Grid) float64 {
				if i >= len(fits) {
					return 0
				}
				v := fits[i]
				i++
				return v
			})
			if err != nil {
				return errObj(err.Error())
			}
			return okObj()
		}))
		obj.Set("bestFitness", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			return pop.BestFitness()
		}))
		obj.Set("summary", js.FuncOf(func(this js.Value, a []js.Value) interface{} {
			gen := 0
			if len(a) > 0 {
				gen = a[0].Int()
			}
			return pop.Summary(gen)
		}))
		return obj
	}))
	js.Global().Set("listWelvetActivations", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr([]string{
			"linear", "relu", "silu", "gelu", "tanh", "sigmoid", "leaky_relu", "relu2",
		})
	}))
	js.Global().Set("listWelvetNativeOnlyCases", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Explicit catalog of Go w2a cases that are host/SIMD/WebGPU honesty — not required on WASM.
		return jsonStr([]map[string]string{
			{"suite": "dense", "case": "SIMD Plan 9 DotTile", "reason": "native-simd"},
			{"suite": "dense", "case": "SIMD Plan 9 enabled on this arch", "reason": "native-simd"},
			{"suite": "dense", "case": "SIMD fused k-cache", "reason": "native-simd"},
			{"suite": "dense", "case": "SIMD fused k/IQ/Affine parity", "reason": "native-simd"},
			{"suite": "dense", "case": "SIMD fused AffinePacked CPU vs SIMD", "reason": "native-simd"},
			{"suite": "*", "case": "Grad verify — CPU vs SIMD agreement", "reason": "native-simd"},
			{"suite": "*", "case": "SIMD FormatNone × all dtypes", "reason": "native-simd"},
			{"suite": "*", "case": "SIMD+WebGPU all quant formats", "reason": "native-simd-webgpu"},
			{"suite": "*", "case": "§12 WebGPU↔CPU parity", "reason": "native-webgpu"},
			{"suite": "*", "case": "WebGPU hard-errors without device", "reason": "native-webgpu"},
			{"suite": "*", "case": "TIMED matrix ns/op tables", "reason": "native-bench"},
			{"suite": "donate", "case": "TCP PutModel + Infer", "reason": "native-net"},
			{"suite": "donate", "case": "Frame round-trip", "reason": "native-net"},
			{"suite": "hardware", "case": "Audit OS/CPU", "reason": "native-host"},
			{"suite": "hardware", "case": "ToJSON parses", "reason": "native-host"},
			{"suite": "fusedgpu", "case": "*", "reason": "native-webgpu"},
			{"suite": "simd", "case": "*", "reason": "native-simd"},
			{"suite": "universal", "case": "LoadUniversal(path)", "reason": "native-fs"},
			{"suite": "memory", "case": "WriteJSON(path)", "reason": "native-fs"},
			{"suite": "serialization", "case": "SaveGridJSON/LoadGridJSON path", "reason": "native-fs"},
		})
	}))
	js.Global().Set("ExtractNetworkBlueprint", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("ExtractNetworkBlueprint(gridId, modelID?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		id := "wasm"
		if len(args) >= 2 {
			id = args[1].String()
		}
		return jsonStr(telemetry.ExtractNetworkBlueprint(g, id))
	}))
	js.Global().Set("CloneGrid", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("CloneGrid(gridId)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		cp, err := evolution.CloneGrid(g)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(cp)
	}))
	js.Global().Set("DefaultTanhiUDPPort", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return tanhi.DefaultUDPPort
	}))
	js.Global().Set("defaultSpliceConfig", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr(evolution.DefaultSpliceConfig())
	}))
	js.Global().Set("defaultNEATConfig", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		d := 64
		if len(args) >= 1 {
			d = args[0].Int()
		}
		return jsonStr(evolution.DefaultNEATConfig(d))
	}))
	js.Global().Set("SpliceDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("SpliceDNA(gridIdA, gridIdB, cfgJSON?)")
		}
		a, okA := getGrid(int64(args[0].Int()))
		b, okB := getGrid(int64(args[1].Int()))
		if !okA || !okB {
			return errObj("invalid grid")
		}
		cfg := evolution.DefaultSpliceConfig()
		if len(args) >= 3 {
			_ = json.Unmarshal([]byte(args[2].String()), &cfg)
		}
		out, err := evolution.SpliceDNA(a, b, cfg)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(out)
	}))
	js.Global().Set("NEATMutate", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return errObj("NEATMutate(gridId, cfgJSON?)")
		}
		g, ok := getGrid(int64(args[0].Int()))
		if !ok {
			return errObj("invalid grid")
		}
		cfg := evolution.DefaultNEATConfig(64)
		if len(args) >= 2 {
			_ = json.Unmarshal([]byte(args[1].String()), &cfg)
		}
		out, err := evolution.NEATMutate(g, cfg)
		if err != nil {
			return errObj(err.Error())
		}
		return createGridWrapper(out)
	}))
}
