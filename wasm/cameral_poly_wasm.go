//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// Cameral poly kinds mirror apps/w2a/suites/parallel/poly_ops.go — dual hemispheres
// of each Op kind inside Stack[Parallel], trained under every named TrainMode.

var cameralPolyKindNames = []string{
	"dense", "mha", "swiglu", "rmsnorm", "layernorm", "softmax", "sequential", "residual",
	"cnn1", "cnn2", "cnn3", "convt1", "convt2", "convt3", "rnn", "lstm",
	"embedding", "kmeans", "mamba", "metacognition", "gdn",
}

func registerCameralPolyGlobals() {
	js.Global().Set("listWelvetCameralPolyKinds", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return jsonStr(cameralPolyKindNames)
	}))
	// TrainCameralPoly(kind, mode, dtype?, format?) → {loss, note} | {error}
	js.Global().Set("TrainCameralPoly", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return errObj("TrainCameralPoly(kind, mode, dtype?, format?)")
		}
		kind := args[0].String()
		modeName := args[1].String()
		dt := core.DTypeFloat32
		format := quant.FormatNone
		if len(args) >= 3 && args[2].Type() == js.TypeNumber {
			dt = core.DType(args[2].Int())
		}
		if len(args) >= 4 && args[3].Type() == js.TypeNumber {
			format = quant.Format(args[3].Int())
		}
		mode, err := parallel.ParseTrainMode(modeName)
		if err != nil {
			return errObj(err.Error())
		}
		loss, note, err := trainCameralPoly(kind, mode, dt, format)
		if err != nil {
			return errObj(err.Error())
		}
		obj := js.Global().Get("Object").New()
		obj.Set("loss", loss)
		obj.Set("note", note)
		obj.Set("mode", mode.String())
		obj.Set("kind", kind)
		return obj
	}))
}

func trainCameralPoly(kind string, mode parallel.TrainMode, dt core.DType, format quant.Format) (loss float64, note string, err error) {
	branches, cfg, x, trainable, err := makeCameralPoly(kind)
	if err != nil {
		return 0, "", err
	}
	if err := stampPolyBranches(branches, dt, format); err != nil {
		return 0, "", err
	}
	s, err := parallel.CameralFromBranches(cfg, branches, nil)
	if err != nil {
		return 0, "", fmt.Errorf("cameral: %w", err)
	}
	s.Exec.Backend = core.BackendCPUTiled
	s.SyncChildExec()
	if kind != "embedding" {
		fillPolyInput(x)
	}
	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		return 0, "", fmt.Errorf("fwd: %w", err)
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.2
	}
	loss, err = parallel.TrainStackMSE(s, x, y, mode, 0.1)
	if err != nil {
		return 0, "", fmt.Errorf("train: %w", err)
	}
	if !trainable {
		return loss, "fwd+train (no trainable stores)", nil
	}
	return loss, "ok", nil
}

func fillPolyInput(t *core.Tensor[float32]) {
	for i := range t.Data {
		t.Data[i] = 0.01 * float32((i%7)+1)
	}
}

type setDTypePacker interface {
	SetDType(core.DType) error
	Pack(quant.Format) error
}

type packOnly interface {
	Pack(quant.Format) error
}

func stampPolyBranches(branches []any, dt core.DType, format quant.Format) error {
	for i, br := range branches {
		if p, ok := br.(setDTypePacker); ok {
			if err := p.SetDType(dt); err != nil {
				return fmt.Errorf("branch %d SetDType: %w", i, err)
			}
			if err := p.Pack(format); err != nil {
				return fmt.Errorf("branch %d Pack: %w", i, err)
			}
			continue
		}
		if p, ok := br.(packOnly); ok && format != quant.FormatNone {
			if err := p.Pack(format); err != nil {
				return fmt.Errorf("branch %d Pack: %w", i, err)
			}
		}
	}
	return nil
}

func makeCameralPoly(kind string) (branches []any, cfg parallel.Config, x *core.Tensor[float32], trainable bool, err error) {
	const dim, seq, batch = 32, 4, 2
	switch kind {
	case "dense":
		a, e1 := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		b, e2 := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "mha":
		mcfg := mha.Config{DModel: dim, NumHeads: 4, MaxSeqLen: seq, Causal: true}
		a, e1 := mha.New(mcfg)
		b, e2 := mha.New(mcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq},
			core.NewTensor[float32](batch, seq, dim), true, nil
	case "swiglu":
		scfg := swiglu.Config{InputDim: dim, IntermediateDim: dim * 2}
		a, e1 := swiglu.New(scfg)
		b, e2 := swiglu.New(scfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "rmsnorm":
		a, e1 := rmsnorm.New(rmsnorm.Config{Dim: dim})
		b, e2 := rmsnorm.New(rmsnorm.Config{Dim: dim})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "layernorm":
		a, e1 := layernorm.New(layernorm.Config{Dim: dim})
		b, e2 := layernorm.New(layernorm.Config{Dim: dim})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "softmax":
		a, e1 := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
		b, e2 := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), false, nil
	case "sequential":
		a, e1 := sequential.New(sequential.Config{Dim: dim, Depth: 2})
		b, e2 := sequential.New(sequential.Config{Dim: dim, Depth: 2})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "residual":
		a, e1 := residual.New(residual.Config{Dim: dim, Depth: 1})
		b, e2 := residual.New(residual.Config{Dim: dim, Depth: 1})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "cnn1":
		ccfg := cnn1.Config{InChannels: 4, Filters: 4, SeqLen: 8, Kernel: 3, Padding: 1}
		a, e1 := cnn1.New(ccfg)
		b, e2 := cnn1.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 8, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 4, 8), true, nil
	case "cnn2":
		ccfg := cnn2.Config{InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
		a, e1 := cnn2.New(ccfg)
		b, e2 := cnn2.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 2, 4, 4), true, nil
	case "cnn3":
		ccfg := cnn3.Config{InChannels: 2, Filters: 2, Depth: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
		a, e1 := cnn3.New(ccfg)
		b, e2 := cnn3.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 2, 2, 4, 4), true, nil
	case "convt1":
		ccfg := convt1.Config{InChannels: 4, Filters: 4, SeqLen: 8, Kernel: 3, Padding: 1}
		a, e1 := convt1.New(ccfg)
		b, e2 := convt1.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 8, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 4, 8), true, nil
	case "convt2":
		ccfg := convt2.Config{InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
		a, e1 := convt2.New(ccfg)
		b, e2 := convt2.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 2, 4, 4), true, nil
	case "convt3":
		ccfg := convt3.Config{InChannels: 2, Filters: 2, Depth: 2, Height: 4, Width: 4, Kernel: 3, Padding: 1}
		a, e1 := convt3.New(ccfg)
		b, e2 := convt3.New(ccfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: 4, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, 2, 2, 4, 4), true, nil
	case "rnn":
		rcfg := rnn.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
		a, e1 := rnn.New(rcfg)
		b, e2 := rnn.New(rcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq},
			core.NewTensor[float32](batch, seq, dim), true, nil
	case "lstm":
		lcfg := lstm.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
		a, e1 := lstm.New(lcfg)
		b, e2 := lstm.New(lcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq},
			core.NewTensor[float32](batch, seq, dim), true, nil
	case "embedding":
		ecfg := embedding.Config{VocabSize: 16, EmbeddingDim: dim, SeqLen: seq}
		a, e1 := embedding.New(ecfg)
		b, e2 := embedding.New(ecfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		x = core.NewTensor[float32](batch, seq)
		for i := range x.Data {
			x.Data[i] = float32(i % 16)
		}
		return []any{a, b}, parallel.Config{Dim: dim, Branches: 2, Combine: parallel.CombineAdd}, x, true, nil
	case "kmeans":
		kcfg := kmeans.Config{NumClusters: 4, FeatureDim: dim, OutputMode: kmeans.OutputFeatures}
		a, e1 := kmeans.New(kcfg)
		b, e2 := kmeans.New(kcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "mamba":
		mcfg := mamba.Config{DModel: dim, DState: 8, SeqLen: seq}
		a, e1 := mamba.New(mcfg)
		b, e2 := mamba.New(mcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq},
			core.NewTensor[float32](batch, seq, dim), true, nil
	case "metacognition":
		a, e1 := metacognition.New(metacognition.Config{Dim: dim})
		b, e2 := metacognition.New(metacognition.Config{Dim: dim})
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd},
			core.NewTensor[float32](batch, dim), true, nil
	case "gdn":
		gcfg := gdn.Config{
			HiddenSize: dim, NumKeyHeads: 2, NumValueHeads: 2,
			KeyHeadDim: 8, ValueHeadDim: 8, ConvKernel: 2, Eps: 1e-6,
		}
		a, e1 := gdn.New(gcfg)
		b, e2 := gdn.New(gcfg)
		if e1 != nil || e2 != nil {
			return nil, cfg, nil, false, firstErr(e1, e2)
		}
		return []any{a, b}, parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq},
			core.NewTensor[float32](batch, seq, dim), true, nil
	default:
		return nil, cfg, nil, false, fmt.Errorf("unknown cameral poly kind %q", kind)
	}
}

func firstErr(a, b error) error {
	if a != nil {
		return a
	}
	return b
}
