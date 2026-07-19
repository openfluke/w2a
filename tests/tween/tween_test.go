package tween_test

import (
	"math"
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/training"
)

func TestTweenGapReduce(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 1)
	// Identity init
	w := make([]float32, 16)
	for i := 0; i < 4; i++ {
		w[i*4+i] = 1
	}
	l0, err := dense.NewConfigured(4, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l0); err != nil {
		t.Fatal(err)
	}

	x := core.NewTensor[float32](1, 4)
	for i := 0; i < 4; i++ {
		x.Data[i] = float32(i + 1)
	}
	target := core.NewTensor[float32](1, 4)
	for i := 0; i < 4; i++ {
		target.Data[i] = float32(i+1) * 2 // want 2x
	}

	loss0, _, err := training.StepTween(g, x, target, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	loss1, st, err := training.StepTween(g, x, target, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(loss0) || math.IsNaN(loss1) {
		t.Fatalf("nan loss %v %v", loss0, loss1)
	}
	if loss1 >= loss0 {
		t.Fatalf("expected loss drop: %v -> %v (gaps=%v)", loss0, loss1, st.Gaps)
	}
}
