package dna_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func TestDNAAcrossDTypeQuant(t *testing.T) {
	dtypes := []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeInt8, core.DTypeBFloat16}
	formats := []quant.Format{quant.FormatNone, quant.FormatQ8_0, quant.FormatQ4_0}

	for _, dt := range dtypes {
		for _, fmt := range formats {
			if fmt != quant.FormatNone && dt != core.DTypeFloat32 {
				// Pack path expects f32 master source; SetDType first then Pack in NewConfigured.
			}
			g := architecture.NewGrid(1, 1, 1, 1)
			init := make([]float32, 16)
			for i := range init {
				init[i] = float32((i%7)-3) * 0.1
			}
			l, err := dense.NewConfigured(4, 4, core.ActivationLinear, dt, fmt, init)
			if err != nil {
				t.Fatalf("dt=%v fmt=%v: %v", dt, fmt, err)
			}
			if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
				t.Fatal(err)
			}
			sig := dna.ExtractDNA(g)
			if len(sig) != 1 || len(sig[0].Weights) == 0 {
				t.Fatalf("dt=%v fmt=%v empty DNA", dt, fmt)
			}
			cmp := dna.CompareNetworks(sig, sig)
			if cmp.OverallOverlap < 0.999 {
				t.Fatalf("dt=%v fmt=%v self-overlap=%v", dt, fmt, cmp.OverallOverlap)
			}
		}
	}
}

func TestDNAMultiLayerOps(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 3)
	d, err := dense.New(8, 8, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	r, err := rmsnorm.New(rmsnorm.Config{Dim: 8})
	if err != nil {
		t.Fatal(err)
	}
	s, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
		t.Fatal(err)
	}
	// Place rmsnorm / swiglu via BindOp
	metaR := r.Core
	metaR.Z, metaR.Y, metaR.X, metaR.L = 0, 0, 0, 1
	if err := g.BindOp(0, 0, 0, 1, metaR, r); err != nil {
		t.Fatal(err)
	}
	metaS := s.Core
	metaS.Z, metaS.Y, metaS.X, metaS.L = 0, 0, 0, 2
	if err := g.BindOp(0, 0, 0, 2, metaS, s); err != nil {
		t.Fatal(err)
	}
	sig := dna.ExtractDNA(g)
	if len(sig) != 3 {
		t.Fatalf("want 3 sigs, got %d", len(sig))
	}
	for i, s := range sig {
		if len(s.Weights) == 0 {
			t.Fatalf("sig[%d] empty", i)
		}
	}
}
