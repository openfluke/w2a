package dna

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Extract+compare immutable smoke (Dense)", Run: extractImmutableSmoke},
		{Name: "Self-compare FormatNone × all 34 dtypes (Dense)", Run: formatNoneAllDTypes},
		{Name: "Self-compare all quants × Float32 (Dense)", Run: allQuantsFloat32},
		{Name: "Multi-layer Ops DNA (Dense/RMS/SwiGLU/MHA/…)", Run: multiLayerOps},
		{Name: "Detect weight drift (cosine < 1 after mutate)", Run: detectDrift},
		{Name: "GAP CENSUS — layer-kind × FormatNone Float32", Run: layerKindCensus},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("dna", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("dna", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("dna: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("dna: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("dna", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("dna", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func rec(op, dt, format, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "dna", Op: op, DType: dt, Format: format, Backend: "cpu", Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func placeDense(g *architecture.Grid, l int, in, out int, dt core.DType, format quant.Format, init []float32) error {
	layer, err := dense.NewConfigured(in, out, core.ActivationLinear, dt, format, init)
	if err != nil {
		return err
	}
	return dense.Place(g, 0, 0, 0, l, layer)
}

func extractImmutableSmoke() error {
	g := architecture.NewGrid(1, 1, 1, 1)
	init := make([]float32, 16)
	for i := range init {
		init[i] = float32(i) * 0.01
	}
	if err := placeDense(g, 0, 4, 4, core.DTypeFloat32, quant.FormatNone, init); err != nil {
		return err
	}
	before, _ := g.At(0, 0, 0, 0).Op.(*dense.Layer).Weights.FlattenF32()
	sig := dna.ExtractDNA(g)
	after, _ := g.At(0, 0, 0, 0).Op.(*dense.Layer).Weights.FlattenF32()
	for i := range before {
		if before[i] != after[i] {
			rec("extract", "f32", "none", "FAIL", "mutated")
			return fmt.Errorf("ExtractDNA mutated weights")
		}
	}
	cmp := dna.CompareNetworks(sig, sig)
	if cmp.OverallOverlap < 0.999 {
		rec("extract", "f32", "none", "FAIL", fmt.Sprintf("overlap=%v", cmp.OverallOverlap))
		return fmt.Errorf("self-overlap %v", cmp.OverallOverlap)
	}
	rec("extract", "f32", "none", "OK", "")
	return nil
}

func formatNoneAllDTypes() error {
	var fails int
	for _, dt := range core.AllDTypes {
		g := architecture.NewGrid(1, 1, 1, 1)
		init := make([]float32, 16)
		for i := range init {
			init[i] = float32((i%7)-3) * 0.05
		}
		if err := placeDense(g, 0, 4, 4, dt, quant.FormatNone, init); err != nil {
			rec("self", dt.String(), "none", "GAP", err.Error())
			fails++
			continue
		}
		sig := dna.ExtractDNA(g)
		cmp := dna.CompareNetworks(sig, sig)
		if len(sig) != 1 || cmp.OverallOverlap < 0.999 {
			rec("self", dt.String(), "none", "FAIL", fmt.Sprintf("overlap=%v", cmp.OverallOverlap))
			fails++
			continue
		}
		rec("self", dt.String(), "none", "OK", "")
	}
	fmt.Printf("(%d FormatNone dtypes) ", len(core.AllDTypes))
	if fails > 0 {
		return fmt.Errorf("%d dtype cells failed/gap", fails)
	}
	return nil
}

func allQuantsFloat32() error {
	var fails, gaps, oks int
	for _, f := range quant.AllFormats {
		g := architecture.NewGrid(1, 1, 1, 1)
		init := make([]float32, 64) // 8×8 — friendlier for block quants
		for i := range init {
			init[i] = float32((i%11)-5) * 0.08
		}
		if err := placeDense(g, 0, 8, 8, core.DTypeFloat32, f, init); err != nil {
			rec("self", "f32", f.String(), "GAP", err.Error())
			gaps++
			continue
		}
		sig := dna.ExtractDNA(g)
		cmp := dna.CompareNetworks(sig, sig)
		if len(sig) != 1 || cmp.OverallOverlap < 0.999 {
			rec("self", "f32", f.String(), "FAIL", fmt.Sprintf("overlap=%v", cmp.OverallOverlap))
			fails++
			continue
		}
		rec("self", "f32", f.String(), "OK", "")
		oks++
	}
	fmt.Printf("(ok=%d gap=%d fail=%d / %d quants) ", oks, gaps, fails, len(quant.AllFormats))
	if fails > 0 {
		return fmt.Errorf("%d quant cells failed", fails)
	}
	return nil
}

func multiLayerOps() error {
	g := architecture.NewGrid(1, 1, 1, 8)
	ops := []struct {
		name string
		bind func(l int) error
	}{
		{"dense", func(l int) error {
			d, err := dense.New(8, 8, core.ActivationLinear, core.DTypeFloat32)
			if err != nil {
				return err
			}
			return dense.Place(g, 0, 0, 0, l, d)
		}},
		{"rmsnorm", func(l int) error {
			r, err := rmsnorm.New(rmsnorm.Config{Dim: 8})
			if err != nil {
				return err
			}
			m := r.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, r)
		}},
		{"swiglu", func(l int) error {
			s, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
			if err != nil {
				return err
			}
			m := s.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, s)
		}},
		{"mha", func(l int) error {
			ml, err := mha.New(mha.Config{DModel: 8, NumHeads: 2, MaxSeqLen: 4})
			if err != nil {
				return err
			}
			m := ml.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, ml)
		}},
		{"layernorm", func(l int) error {
			ln, err := layernorm.New(layernorm.Config{Dim: 8})
			if err != nil {
				return err
			}
			m := ln.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, ln)
		}},
		{"softmax", func(l int) error {
			sm, err := softmax.New(softmax.Config{Dim: 8, SeqLen: 4})
			if err != nil {
				return err
			}
			m := sm.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, sm)
		}},
		{"sequential", func(l int) error {
			sq, err := sequential.New(sequential.Config{Dim: 8, Depth: 2, SeqLen: 4})
			if err != nil {
				return err
			}
			m := sq.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, sq)
		}},
		{"residual", func(l int) error {
			rs, err := residual.New(residual.Config{Dim: 8, Depth: 1, SeqLen: 4})
			if err != nil {
				return err
			}
			m := rs.Core
			m.Z, m.Y, m.X, m.L = 0, 0, 0, l
			return g.BindOp(0, 0, 0, l, m, rs)
		}},
	}
	for i, op := range ops {
		if err := op.bind(i); err != nil {
			rec("multilayer", op.name, "none", "FAIL", err.Error())
			return err
		}
		rec("multilayer", op.name, "none", "OK", "")
	}
	sig := dna.ExtractDNA(g)
	if len(sig) != len(ops) {
		return fmt.Errorf("want %d signatures, got %d", len(ops), len(sig))
	}
	cmp := dna.CompareNetworks(sig, sig)
	if cmp.OverallOverlap < 0.999 {
		return fmt.Errorf("multi self-overlap %v", cmp.OverallOverlap)
	}
	return nil
}

func detectDrift() error {
	g1 := architecture.NewGrid(1, 1, 1, 1)
	g2 := architecture.NewGrid(1, 1, 1, 1)
	init := make([]float32, 16)
	for i := range init {
		init[i] = float32(i+1) * 0.05 // non-uniform — cosine needs shape change
	}
	if err := placeDense(g1, 0, 4, 4, core.DTypeFloat32, quant.FormatNone, init); err != nil {
		return err
	}
	mut := append([]float32(nil), init...)
	for i := range mut {
		if i%2 == 0 {
			mut[i] = -mut[i]
		} else {
			mut[i] *= 0.25
		}
	}
	if err := placeDense(g2, 0, 4, 4, core.DTypeFloat32, quant.FormatNone, mut); err != nil {
		return err
	}
	cmp := dna.CompareNetworks(dna.ExtractDNA(g1), dna.ExtractDNA(g2))
	if cmp.OverallOverlap > 0.999 {
		rec("drift", "f32", "none", "FAIL", "no drift detected")
		return fmt.Errorf("expected drift, overlap=%v", cmp.OverallOverlap)
	}
	rec("drift", "f32", "none", "OK", fmt.Sprintf("overlap=%.4f", cmp.OverallOverlap))
	return nil
}

func layerKindCensus() error {
	kinds := []struct {
		name string
		mk   func() (any, core.Layer, error)
	}{
		{"dense", func() (any, core.Layer, error) {
			l, err := dense.New(8, 8, core.ActivationLinear, core.DTypeFloat32)
			return l, l.Core, err
		}},
		{"rmsnorm", func() (any, core.Layer, error) {
			l, err := rmsnorm.New(rmsnorm.Config{Dim: 8})
			return l, l.Core, err
		}},
		{"layernorm", func() (any, core.Layer, error) {
			l, err := layernorm.New(layernorm.Config{Dim: 8})
			return l, l.Core, err
		}},
		{"swiglu", func() (any, core.Layer, error) {
			l, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
			return l, l.Core, err
		}},
		{"mha", func() (any, core.Layer, error) {
			l, err := mha.New(mha.Config{DModel: 8, NumHeads: 2, MaxSeqLen: 4})
			return l, l.Core, err
		}},
		{"cnn1", func() (any, core.Layer, error) {
			l, err := cnn1.New(cnn1.Config{InChannels: 1, Filters: 2, SeqLen: 8, Kernel: 3, Activation: core.ActivationLinear})
			return l, l.Core, err
		}},
		{"rnn", func() (any, core.Layer, error) {
			l, err := rnn.New(rnn.Config{InputSize: 4, HiddenSize: 4, SeqLen: 2})
			return l, l.Core, err
		}},
		{"lstm", func() (any, core.Layer, error) {
			l, err := lstm.New(lstm.Config{InputSize: 4, HiddenSize: 4, SeqLen: 2})
			return l, l.Core, err
		}},
		{"embedding", func() (any, core.Layer, error) {
			l, err := embedding.New(embedding.Config{VocabSize: 16, EmbeddingDim: 8, SeqLen: 4})
			return l, l.Core, err
		}},
		{"softmax", func() (any, core.Layer, error) {
			l, err := softmax.New(softmax.Config{Dim: 8, SeqLen: 4})
			return l, l.Core, err
		}},
		{"sequential", func() (any, core.Layer, error) {
			l, err := sequential.New(sequential.Config{Dim: 8, Depth: 2, SeqLen: 4})
			return l, l.Core, err
		}},
		{"residual", func() (any, core.Layer, error) {
			l, err := residual.New(residual.Config{Dim: 8, Depth: 1, SeqLen: 4})
			return l, l.Core, err
		}},
	}
	var gaps, fails, oks int
	for _, k := range kinds {
		g := architecture.NewGrid(1, 1, 1, 1)
		op, meta, err := k.mk()
		if err != nil {
			rec("census", k.name, "none", "GAP", err.Error())
			gaps++
			continue
		}
		meta.Z, meta.Y, meta.X, meta.L = 0, 0, 0, 0
		if err := g.BindOp(0, 0, 0, 0, meta, op); err != nil {
			rec("census", k.name, "none", "FAIL", err.Error())
			fails++
			continue
		}
		sig := dna.ExtractDNA(g)
		if len(sig) != 1 || len(sig[0].Weights) == 0 {
			rec("census", k.name, "none", "FAIL", "empty DNA")
			fails++
			continue
		}
		cmp := dna.CompareNetworks(sig, sig)
		if cmp.OverallOverlap < 0.999 {
			rec("census", k.name, "none", "FAIL", fmt.Sprintf("overlap=%v", cmp.OverallOverlap))
			fails++
			continue
		}
		rec("census", k.name, "none", "OK", "")
		oks++
	}
	fmt.Printf("(ok=%d gap=%d fail=%d / %d kinds) ", oks, gaps, fails, len(kinds))
	if fails > 0 {
		return fmt.Errorf("%d layer-kind DNA failures", fails)
	}
	return nil
}
