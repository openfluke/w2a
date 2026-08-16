package residual_test

import (
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/systems/dna"
)

func TestNewFromOpsRMSSwiGLU(t *testing.T) {
	const dim = 8
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
	l, err := residual.NewFromOps(residual.Config{Dim: dim}, []any{rms, ln, sw})
	if err != nil {
		t.Fatal(err)
	}
	if len(l.ChildOps()) != 3 {
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
	pre, post, err := residual.Forward(l, x)
	if err != nil {
		t.Fatal(err)
	}
	if post.Len() != x.Len() {
		t.Fatalf("skip shape %v vs %v", post.Shape, x.Shape)
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := residual.Backward(l, gy, x, pre)
	if err != nil {
		t.Fatal(err)
	}
	if err := residual.ApplyGradSGD(l, dW, 0.1); err != nil {
		t.Fatal(err)
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		t.Fatal(err)
	}
	moved := false
	for i := range before {
		if before[i] != after[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("mixed residual F weights unchanged")
	}
}
