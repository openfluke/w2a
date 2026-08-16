package metacognition_test

import (
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/quant"
)

func TestBackwardNilPreDoesNotPanic(t *testing.T) {
	l, err := metacognition.NewConfigured[float32](metacognition.Config{Dim: 8}, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, 8)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	_, post, err := metacognition.Forward(l, x)
	if err != nil {
		t.Fatal(err)
	}
	gy := core.NewTensor[float32](post.Shape...)
	copy(gy.Data, post.Data)
	gx, dW, err := metacognition.Backward(l, gy, x, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gx == nil || gx.Len() != x.Len() {
		t.Fatalf("gx %v", gx)
	}
	if dW == nil || dW.Len() == 0 {
		t.Fatal("nil dW")
	}
}
