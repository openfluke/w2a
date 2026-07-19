package densenet_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/quant"
)

// Two Dense cells in a 1×1×1×2 volumetric stack: identity → scale, then backward.
func TestVolumetricDenseChain(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 2)
	g.Exec.Backend = core.BackendCPUTiled

	const in, mid, out = 4, 4, 4
	id := make([]float32, mid*in)
	for i := 0; i < mid; i++ {
		id[i*in+i] = 1
	}
	scale := make([]float32, out*mid)
	for i := 0; i < out; i++ {
		scale[i*mid+i] = 2
	}

	l0, err := dense.NewConfigured(in, mid, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, id)
	if err != nil {
		t.Fatal(err)
	}
	l1, err := dense.NewConfigured(mid, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, scale)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l0); err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 1, l1); err != nil {
		t.Fatal(err)
	}

	x := core.NewTensor[float32](1, in)
	for i := 0; i < in; i++ {
		x.Data[i] = float32(i + 1) // 1,2,3,4
	}

	fwd, err := forward.Forward(g, x)
	if err != nil {
		t.Fatal(err)
	}
	if len(fwd.Steps) != 2 {
		t.Fatalf("want 2 steps, got %d", len(fwd.Steps))
	}
	// identity then ×2 → output = 2,4,6,8
	want := []float32{2, 4, 6, 8}
	for i, w := range want {
		if fwd.Output.Data[i] != w {
			t.Fatalf("out[%d]=%v want %v (full=%v)", i, fwd.Output.Data[i], w, fwd.Output.Data)
		}
	}

	gy := core.NewTensor[float32](1, out)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		t.Fatal(err)
	}
	if bwd.GradIn == nil || len(bwd.GradWs) != 2 {
		t.Fatalf("bwd incomplete: gin=%v dWs=%d", bwd.GradIn, len(bwd.GradWs))
	}
	// dL/dx through ×2 then I → gradIn = 2,2,2,2
	for i := 0; i < in; i++ {
		if bwd.GradIn.Data[i] != 2 {
			t.Fatalf("gradIn[%d]=%v want 2", i, bwd.GradIn.Data[i])
		}
	}
}

func TestVolumetricRemoteHop(t *testing.T) {
	// 1×1×2×1: cell A at x=0 produces, cell B at x=1 is remote-linked to A
	// (skips sequential input from… there is no prior sequential if B is second).
	g := architecture.NewGrid(1, 1, 2, 1)
	g.Exec.Backend = core.BackendCPUTiled

	const n = 3
	id := make([]float32, n*n)
	for i := 0; i < n; i++ {
		id[i*n+i] = 1
	}
	a, err := dense.NewConfigured(n, n, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, id)
	if err != nil {
		t.Fatal(err)
	}
	b, err := dense.NewConfigured(n, n, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, id)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, a); err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 1, 0, b); err != nil {
		t.Fatal(err)
	}
	// B takes activation from A (same as sequential here, but exercises hop path).
	if err := g.SetRemoteLink(0, 0, 1, 0, 0, 0, 0, 0); err != nil {
		t.Fatal(err)
	}

	x := core.NewTensor[float32](1, n)
	for i := 0; i < n; i++ {
		x.Data[i] = float32(i + 1)
	}
	fwd, err := forward.Forward(g, x)
	if err != nil {
		t.Fatal(err)
	}
	if len(fwd.Steps) != 2 {
		t.Fatalf("steps=%d", len(fwd.Steps))
	}
	// I @ I @ x = x
	for i := 0; i < n; i++ {
		if fwd.Output.Data[i] != x.Data[i] {
			t.Fatalf("out=%v want %v", fwd.Output.Data, x.Data)
		}
	}
}
