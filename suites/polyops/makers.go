// Package polyops builds single-op grids for DNA / evolution / tween matrix suites.
package polyops

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// Kind is one implemented Op covered by the poly system matrices.
type Kind struct {
	Name string
	// Make builds a 1×1×1×1 grid with one Op at (0,0,0,0).
	Make func(dt core.DType, format quant.Format) (*architecture.Grid, error)
	// InputShape / TargetShape are for tween StepTween (float32 acts).
	InputShape  []int
	TargetShape []int
	// FillInput optional — e.g. embedding token IDs.
	FillInput func(x *core.Tensor[float32])
}

// AllKinds — every implemented weighted Op (+ weightless Softmax).
func AllKinds() []Kind {
	const dim = 16
	return []Kind{
		{
			Name: "dense",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				init := ramp(dim * dim)
				l, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, format, init)
				if err != nil {
					return nil, err
				}
				g := architecture.NewGrid(1, 1, 1, 1)
				return g, dense.Place(g, 0, 0, 0, 0, l)
			},
			InputShape:  []int{1, dim},
			TargetShape: []int{1, dim},
		},
		{
			Name: "rmsnorm",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				ones := onesF(dim)
				l, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim}, dt, format, ones)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, dim},
			TargetShape: []int{1, dim},
		},
		{
			Name: "layernorm",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				ones, zeros := onesF(dim), make([]float32, dim)
				l, err := layernorm.NewConfigured(layernorm.Config{Dim: dim}, dt, format, ones, zeros)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, dim},
			TargetShape: []int{1, dim},
		},
		{
			Name: "swiglu",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := swiglu.NewConfigured[float32](swiglu.Config{InputDim: dim, IntermediateDim: dim * 2}, dt, format, nil, nil, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, dim},
			TargetShape: []int{1, dim},
		},
		{
			Name: "mha",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := mha.NewConfigured[float32](mha.Config{DModel: dim, NumHeads: 2, MaxSeqLen: 4}, dt, format, nil, nil, nil, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 4, dim},
			TargetShape: []int{1, 4, dim},
		},
		{
			Name: "cnn1",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				cfg := cnn1.Config{InChannels: 2, Filters: 2, SeqLen: 8, Kernel: 3, Padding: 1, Activation: core.ActivationLinear}
				l, err := cnn1.NewConfigured[float32](cfg, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, 8},
			TargetShape: []int{1, 2, 8}, // padding=1 → same length
		},
		{
			Name: "cnn2",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				cfg := cnn2.Config{InChannels: 2, Filters: 2, Height: 6, Width: 6, Kernel: 3, Padding: 1, Activation: core.ActivationLinear}
				l, err := cnn2.NewConfigured[float32](cfg, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, 6, 6},
			TargetShape: []int{1, 2, 6, 6},
		},
		{
			Name: "cnn3",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				cfg := cnn3.Config{InChannels: 1, Filters: 1, Depth: 4, Height: 4, Width: 4, Kernel: 3, Padding: 1, Activation: core.ActivationLinear}
				l, err := cnn3.NewConfigured[float32](cfg, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 1, 4, 4, 4},
			TargetShape: []int{1, 1, 4, 4, 4},
		},
		{
			Name: "rnn",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := rnn.NewConfigured[float32](rnn.Config{InputSize: 8, HiddenSize: 8, SeqLen: 2}, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, 8},
			TargetShape: []int{1, 2, 8},
		},
		{
			Name: "lstm",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := lstm.NewConfigured[float32](lstm.Config{InputSize: 8, HiddenSize: 8, SeqLen: 2}, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, 8},
			TargetShape: []int{1, 2, 8},
		},
		{
			Name: "embedding",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				cfg := embedding.Config{VocabSize: 32, EmbeddingDim: dim, SeqLen: 4}
				l, err := embedding.NewConfigured(cfg, dt, format, ramp(cfg.WeightCount()))
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 4},
			TargetShape: []int{1, 4, dim},
			FillInput: func(x *core.Tensor[float32]) {
				for i := range x.Data {
					x.Data[i] = float32(i % 32)
				}
			},
		},
		{
			Name: "sequential",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := sequential.NewConfigured[float32](sequential.Config{Dim: dim, Depth: 2, SeqLen: 2}, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, dim},
			TargetShape: []int{1, 2, dim},
		},
		{
			Name: "residual",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := residual.NewConfigured[float32](residual.Config{Dim: dim, Depth: 1, SeqLen: 2}, dt, format, nil)
				if err != nil {
					return nil, err
				}
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, dim},
			TargetShape: []int{1, 2, dim},
		},
		{
			Name: "softmax",
			Make: func(dt core.DType, format quant.Format) (*architecture.Grid, error) {
				l, err := softmax.New(softmax.Config{Dim: dim, SeqLen: 2})
				if err != nil {
					return nil, err
				}
				_ = l.SetDType(dt)
				_ = l.Pack(format) // no-op
				return bind(l.Core, l)
			},
			InputShape:  []int{1, 2, dim},
			TargetShape: []int{1, 2, dim},
		},
	}
}

func bind(meta core.Layer, op any) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	meta.Z, meta.Y, meta.X, meta.L = 0, 0, 0, 0
	if err := g.BindOp(0, 0, 0, 0, meta, op); err != nil {
		return nil, err
	}
	return g, nil
}

func ramp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%11)-5) * 0.05
	}
	return out
}

func onesF(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// MakeIO builds float32 input/target for tween on a Kind.
func MakeIO(k Kind, scale float64) (x, target *core.Tensor[float32]) {
	x = core.NewTensor[float32](k.InputShape...)
	target = core.NewTensor[float32](k.TargetShape...)
	if k.FillInput != nil {
		k.FillInput(x)
	} else {
		for i := range x.Data {
			x.Data[i] = float32((i%7)+1) * 0.1
		}
	}
	for i := range target.Data {
		target.Data[i] = float32(float64((i%7)+1)*0.1*scale)
	}
	return x, target
}

// Summary formats ok/gap/fail counts.
func Summary(ok, gap, fail, total int) string {
	return fmt.Sprintf("(ok=%d gap=%d fail=%d / %d)", ok, gap, fail, total)
}
