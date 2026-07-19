package dna_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

func TestExtractCompareImmutable(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 2)
	w0 := make([]float32, 16)
	w1 := make([]float32, 16)
	for i := range w0 {
		w0[i] = float32(i) * 0.01
		w1[i] = float32(i) * 0.02
	}
	l0, err := dense.NewConfigured(4, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w0)
	if err != nil {
		t.Fatal(err)
	}
	l1, err := dense.NewConfigured(4, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w1)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l0); err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 1, l1); err != nil {
		t.Fatal(err)
	}

	before, err := l0.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	d := dna.ExtractDNA(g)
	if len(d) != 2 {
		t.Fatalf("want 2 signatures, got %d", len(d))
	}
	after, err := l0.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("ExtractDNA mutated weights at %d", i)
		}
	}

	cmp := dna.CompareNetworks(d, d)
	if cmp.OverallOverlap < 0.999 {
		t.Fatalf("self-compare overlap=%v", cmp.OverallOverlap)
	}
}
