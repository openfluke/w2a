package dense_test

import (
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

func TestLinearGradInAndGradWOnly(t *testing.T) {
	l, err := dense.NewConfigured[float32](4, 3, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	w, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i := range w {
		w[i] = 0.1
	}
	if err := l.Weights.SetFromF32(w); err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, 4)
	for i := range x.Data {
		x.Data[i] = 0.5
	}
	pre, _, err := dense.Forward(l, x)
	if err != nil {
		t.Fatal(err)
	}
	gy := core.NewTensor[float32](1, 3)
	gy.Data[0], gy.Data[1], gy.Data[2] = 1, -0.5, 0.25
	gx, err := dense.LinearGradIn(l, gy, x)
	if err != nil {
		t.Fatal(err)
	}
	if gx.Len() != 4 {
		t.Fatalf("gx len %d", gx.Len())
	}
	dW, err := dense.GradWOnly(l, gy, x, pre)
	if err != nil {
		t.Fatal(err)
	}
	if dW.Len() != 3*4 {
		t.Fatalf("dW len %d", dW.Len())
	}
	ok := false
	for _, v := range dW.Data {
		if v != 0 {
			ok = true
			break
		}
	}
	if !ok {
		t.Fatal("GradWOnly was all zeros")
	}
	dW2, err := dense.GradWOnly(l, gy, x, nil)
	if err != nil {
		t.Fatal(err)
	}
	if dW2 == nil || dW2.Len() != dW.Len() {
		t.Fatalf("GradWOnly nil-pre len %v", dW2)
	}
}
