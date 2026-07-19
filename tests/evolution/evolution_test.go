package evolution_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/evolution"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

func placeDense(t *testing.T, g *architecture.Grid, l int, w []float32) {
	t.Helper()
	layer, err := dense.NewConfigured(4, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, l, layer); err != nil {
		t.Fatal(err)
	}
}

func TestSpliceBlend(t *testing.T) {
	a := architecture.NewGrid(1, 1, 1, 1)
	b := architecture.NewGrid(1, 1, 1, 1)
	wa := make([]float32, 16)
	wb := make([]float32, 16)
	for i := range wa {
		wa[i] = 1
		wb[i] = 3
	}
	placeDense(t, a, 0, wa)
	placeDense(t, b, 0, wb)

	cfg := evolution.DefaultSpliceConfig()
	cfg.BlendAlpha = 0.5
	child, err := evolution.SpliceDNA(a, b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	sig := dna.ExtractDNA(child)
	if len(sig) != 1 || len(sig[0].Weights) == 0 {
		t.Fatalf("bad child DNA %+v", sig)
	}
	// Flattened child weights should be midpoints before normalize — check store.
	cell := child.At(0, 0, 0, 0)
	dl := cell.Op.(*dense.Layer)
	flat, err := dl.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range flat {
		if v < 1.9 || v > 2.1 {
			t.Fatalf("child[%d]=%v want ~2", i, v)
		}
	}
}
