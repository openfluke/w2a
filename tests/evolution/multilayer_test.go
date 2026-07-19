package evolution_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/evolution"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func TestCloneSpliceMultiLayer(t *testing.T) {
	mk := func() *architecture.Grid {
		g := architecture.NewGrid(1, 1, 1, 3)
		d, err := dense.New(8, 8, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			t.Fatal(err)
		}
		if err := d.Weights.Pack(quant.FormatQ8_0); err != nil {
			t.Fatal(err)
		}
		r, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: 8}, core.DTypeFloat16, quant.FormatNone, []float32{1, 1, 1, 1, 1, 1, 1, 1})
		if err != nil {
			t.Fatal(err)
		}
		s, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Pack(quant.FormatQ4_0); err != nil {
			t.Fatal(err)
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
			t.Fatal(err)
		}
		mr := r.Core
		mr.Z, mr.Y, mr.X, mr.L = 0, 0, 0, 1
		_ = g.BindOp(0, 0, 0, 1, mr, r)
		ms := s.Core
		ms.Z, ms.Y, ms.X, ms.L = 0, 0, 0, 2
		_ = g.BindOp(0, 0, 0, 2, ms, s)
		return g
	}
	a := mk()
	b := mk()
	// Perturb B dense weights
	cell := b.At(0, 0, 0, 0)
	dl := cell.Op.(*dense.Layer)
	w, err := dl.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i := range w {
		w[i] += 0.1
	}
	if err := dl.Weights.SetFromF32(w); err != nil {
		t.Fatal(err)
	}

	child, err := evolution.SpliceDNA(a, b, evolution.DefaultSpliceConfig())
	if err != nil {
		t.Fatal(err)
	}
	sig := dna.ExtractDNA(child)
	if len(sig) != 3 {
		t.Fatalf("child DNA len=%d", len(sig))
	}
	// Clone round-trip preserves formats
	clone, err := evolution.CloneGrid(a)
	if err != nil {
		t.Fatal(err)
	}
	ca := dna.ExtractDNA(a)
	cc := dna.ExtractDNA(clone)
	cmp := dna.CompareNetworks(ca, cc)
	if cmp.OverallOverlap < 0.999 {
		t.Fatalf("clone DNA drift overlap=%v", cmp.OverallOverlap)
	}
}
