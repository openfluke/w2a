package tanhi_test

import (
	"testing"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/tanhi"
)

func TestEmitDisabledNoPanic(t *testing.T) {
	cfg := &tanhi.UDPConfig{Enabled: false}
	tanhi.EmitSweep(cfg, "unused")
	tanhi.Emit(cfg, "fwd", 0, &architecture.Cell{Layer: core.Layer{Type: core.LayerDense}}, time.Now(), time.Now(), nil)

	g := architecture.NewGrid(1, 1, 1, 1)
	g.Tanhi = &tanhi.UDPConfig{Enabled: false, SendShape: true}
	l0, err := dense.NewConfigured(2, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l0); err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, 2)
	x.Data[0], x.Data[1] = 1, 2
	if _, err := forward.Forward(g, x); err != nil {
		t.Fatal(err)
	}
}
