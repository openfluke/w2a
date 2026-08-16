package sequential_test

import (
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

func TestNewFromOpsDenseRMSSwiGLU(t *testing.T) {
	const dim = 8
	d, err := dense.NewConfigured[float32](dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	rms, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		t.Fatal(err)
	}
	ln, err := layernorm.New(layernorm.Config{Dim: dim})
	if err != nil {
		t.Fatal(err)
	}
	sw, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		t.Fatal(err)
	}
	l, err := sequential.NewFromOps(sequential.Config{Dim: dim}, []any{d, rms, ln, sw})
	if err != nil {
		t.Fatal(err)
	}
	if len(l.ChildOps()) != 4 {
		t.Fatalf("ops %d", len(l.ChildOps()))
	}
	for _, s := range dna.CollectStores(l) {
		if s == nil {
			continue
		}
		n := s.Rows * s.Cols
		w := make([]float32, n)
		for i := range w {
			w[i] = 0.05
		}
		if s.Rows == s.Cols {
			for i := 0; i < s.Rows; i++ {
				w[i*s.Cols+i] = 1
			}
		}
		_ = s.SetFromF32(w)
	}
	before, err := dna.FlattenOp(l)
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](2, dim)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	pre, post, err := sequential.Forward(l, x)
	if err != nil {
		t.Fatal(err)
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := sequential.Backward(l, gy, x, pre)
	if err != nil {
		t.Fatal(err)
	}
	if err := sequential.ApplyGradSGD(l, dW, 0.1); err != nil {
		t.Fatal(err)
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 || len(before) != len(after) {
		t.Fatalf("flat %d → %d", len(before), len(after))
	}
	moved := false
	for i := range before {
		if before[i] != after[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("mixed sequential weights unchanged")
	}
}
