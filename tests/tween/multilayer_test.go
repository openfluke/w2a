package tween_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/tween"
)

func TestTweenMultiLayerChainRule(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 2)
	d, err := dense.New(4, 4, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	// identity-ish
	w := make([]float32, 16)
	for i := 0; i < 4; i++ {
		w[i*4+i] = 1
	}
	_ = d.Weights.SetFromF32(w)
	r, err := rmsnorm.New(rmsnorm.Config{Dim: 4})
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
		t.Fatal(err)
	}
	mr := r.Core
	mr.Z, mr.Y, mr.X, mr.L = 0, 0, 0, 1
	if err := g.BindOp(0, 0, 0, 1, mr, r); err != nil {
		t.Fatal(err)
	}

	x := core.NewTensor[float32](1, 4)
	for i := 0; i < 4; i++ {
		x.Data[i] = float32(i + 1)
	}
	target := core.NewTensor[float32](1, 4)
	for i := 0; i < 4; i++ {
		target.Data[i] = float32(i+1) * 1.5
	}

	fwd, err := forward.Forward(g, x)
	if err != nil {
		t.Fatal(err)
	}
	st, err := training.ApplyTween(g, fwd, x, target, 0.02)
	if err != nil {
		t.Fatal(err)
	}
	if st == nil {
		t.Fatal("nil tween state")
	}
	// Layerwise path also works
	cfg := tween.DefaultConfig()
	cfg.UseChainRule = false
	st2 := tween.NewState[float32](g, cfg)
	tween.CaptureFromForward(st2, fwd, x)
	if err := tween.Backward(g, st2, target); err != nil {
		t.Fatal(err)
	}
	st2.CalculateLinkBudgets()
	if err := tween.ApplyGaps(g, st2, 0.02); err != nil {
		t.Fatal(err)
	}
}
