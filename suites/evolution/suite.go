package evolution

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/systems/evolution"
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
		{Name: "Splice blend smoke (Dense)", Run: spliceBlendSmoke},
		{Name: "NEAT mutate + population one-gen smoke", Run: neatPopulationSmoke},
		{Name: "Multi-layer clone+splice (Dense/RMS/SwiGLU + quant)", Run: multiLayerCloneSplice},
		{Name: "MATRIX — FormatNone × all 34 dtypes × all layers (clone+splice)", Run: MatrixFormatNoneAllDTypes},
		{Name: "MATRIX — all quants × Float32 × all layers (clone+splice)", Run: MatrixAllQuantsFloat32},
		{Name: "FULL CENSUS — all layers × all dtypes × all quants", Run: FullMatrixCensus},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("evolution", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("evolution", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("evolution: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("evolution: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("evolution", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("evolution", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func rec(op, dt, format, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "evolution", Op: op, DType: dt, Format: format, Backend: "cpu", Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func placeDenseGrid(dt core.DType, format quant.Format, init []float32) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	in, out := 4, 4
	if len(init) == 64 {
		in, out = 8, 8
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, dt, format, init)
	if err != nil {
		return nil, err
	}
	if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
		return nil, err
	}
	return g, nil
}

func spliceBlendSmoke() error {
	wa := make([]float32, 16)
	wb := make([]float32, 16)
	for i := range wa {
		wa[i], wb[i] = 1, 3
	}
	a, err := placeDenseGrid(core.DTypeFloat32, quant.FormatNone, wa)
	if err != nil {
		return err
	}
	b, err := placeDenseGrid(core.DTypeFloat32, quant.FormatNone, wb)
	if err != nil {
		return err
	}
	cfg := evolution.DefaultSpliceConfig()
	cfg.BlendAlpha = 0.5
	child, err := evolution.SpliceDNA(a, b, cfg)
	if err != nil {
		rec("splice", "f32", "none", "FAIL", err.Error())
		return err
	}
	flat, err := child.At(0, 0, 0, 0).Op.(*dense.Layer).Weights.FlattenF32()
	if err != nil {
		return err
	}
	for _, v := range flat {
		if v < 1.9 || v > 2.1 {
			rec("splice", "f32", "none", "FAIL", fmt.Sprintf("got %v", v))
			return fmt.Errorf("blend want ~2, got %v", v)
		}
	}
	rec("splice", "f32", "none", "OK", "")
	return nil
}

func cloneAllOps() error {
	makers := []struct {
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
	var fails int
	for _, m := range makers {
		g := architecture.NewGrid(1, 1, 1, 1)
		op, meta, err := m.mk()
		if err != nil {
			rec("clone", m.name, "none", "GAP", err.Error())
			fails++
			continue
		}
		meta.Z, meta.Y, meta.X, meta.L = 0, 0, 0, 0
		if err := g.BindOp(0, 0, 0, 0, meta, op); err != nil {
			rec("clone", m.name, "none", "FAIL", err.Error())
			fails++
			continue
		}
		clone, err := evolution.CloneGrid(g)
		if err != nil {
			rec("clone", m.name, "none", "FAIL", err.Error())
			fails++
			continue
		}
		cmp := dna.CompareNetworks(dna.ExtractDNA(g), dna.ExtractDNA(clone))
		if cmp.OverallOverlap < 0.999 {
			rec("clone", m.name, "none", "FAIL", fmt.Sprintf("overlap=%v", cmp.OverallOverlap))
			fails++
			continue
		}
		rec("clone", m.name, "none", "OK", "")
	}
	fmt.Printf("(%d ops) ", len(makers))
	if fails > 0 {
		return fmt.Errorf("%d clone failures", fails)
	}
	return nil
}

func spliceAllDTypes() error {
	var fails int
	for _, dt := range core.AllDTypes {
		init := make([]float32, 16)
		for i := range init {
			init[i] = float32((i%5)+1) * 0.05
		}
		a, err := placeDenseGrid(dt, quant.FormatNone, init)
		if err != nil {
			rec("splice", dt.String(), "none", "GAP", err.Error())
			fails++
			continue
		}
		bInit := append([]float32(nil), init...)
		for i := range bInit {
			bInit[i] *= 2
		}
		b, err := placeDenseGrid(dt, quant.FormatNone, bInit)
		if err != nil {
			rec("splice", dt.String(), "none", "GAP", err.Error())
			fails++
			continue
		}
		if _, err := evolution.SpliceDNA(a, b, evolution.DefaultSpliceConfig()); err != nil {
			rec("splice", dt.String(), "none", "FAIL", err.Error())
			fails++
			continue
		}
		rec("splice", dt.String(), "none", "OK", "")
	}
	fmt.Printf("(%d dtypes) ", len(core.AllDTypes))
	if fails > 0 {
		return fmt.Errorf("%d dtype splice fails/gaps", fails)
	}
	return nil
}

func spliceAllQuants() error {
	var fails, gaps, oks int
	for _, f := range quant.AllFormats {
		init := make([]float32, 64)
		for i := range init {
			init[i] = float32((i%9)-4) * 0.1
		}
		a, err := placeDenseGrid(core.DTypeFloat32, f, init)
		if err != nil {
			rec("splice", "f32", f.String(), "GAP", err.Error())
			gaps++
			continue
		}
		bInit := append([]float32(nil), init...)
		for i := range bInit {
			bInit[i] += 0.2
		}
		b, err := placeDenseGrid(core.DTypeFloat32, f, bInit)
		if err != nil {
			rec("splice", "f32", f.String(), "GAP", err.Error())
			gaps++
			continue
		}
		child, err := evolution.SpliceDNA(a, b, evolution.DefaultSpliceConfig())
		if err != nil {
			rec("splice", "f32", f.String(), "FAIL", err.Error())
			fails++
			continue
		}
		// Format preserved on child store
		ws := child.At(0, 0, 0, 0).Op.(*dense.Layer).Weights
		if ws.Format != f {
			rec("splice", "f32", f.String(), "FAIL", fmt.Sprintf("format %s != %s", ws.Format, f))
			fails++
			continue
		}
		rec("splice", "f32", f.String(), "OK", "")
		oks++
	}
	fmt.Printf("(ok=%d gap=%d fail=%d / %d quants) ", oks, gaps, fails, len(quant.AllFormats))
	if fails > 0 {
		return fmt.Errorf("%d quant splice fails", fails)
	}
	return nil
}

func neatPopulationSmoke() error {
	init := make([]float32, 16)
	for i := range init {
		init[i] = 0.1
	}
	seed, err := placeDenseGrid(core.DTypeFloat32, quant.FormatNone, init)
	if err != nil {
		return err
	}
	cfg := evolution.DefaultNEATConfig(4)
	cfg.Seed = 42
	pop, err := evolution.NewNEATPopulation(seed, 4, cfg)
	if err != nil {
		rec("neat", "f32", "none", "FAIL", err.Error())
		return err
	}
	if err := pop.Evolve(func(g *architecture.Grid) float64 {
		sig := dna.ExtractDNA(g)
		if len(sig) == 0 {
			return 0
		}
		var s float64
		for _, w := range sig[0].Weights {
			s += float64(w * w)
		}
		return s
	}); err != nil {
		rec("neat", "f32", "none", "FAIL", err.Error())
		return err
	}
	if pop.Best() == nil {
		return fmt.Errorf("nil best")
	}
	rec("neat", "f32", "none", "OK", pop.Summary(1))
	return nil
}

func multiLayerCloneSplice() error {
	mk := func(perturb bool) (*architecture.Grid, error) {
		g := architecture.NewGrid(1, 1, 1, 3)
		d, err := dense.New(8, 8, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		if err := d.Weights.Pack(quant.FormatQ8_0); err != nil {
			return nil, err
		}
		if perturb {
			w, _ := d.Weights.FlattenF32()
			for i := range w {
				w[i] += 0.15
			}
			_ = d.Weights.SetFromF32(w)
		}
		r, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: 8}, core.DTypeFloat16, quant.FormatNone, []float32{1, 1, 1, 1, 1, 1, 1, 1})
		if err != nil {
			return nil, err
		}
		s, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
		if err != nil {
			return nil, err
		}
		if err := s.Pack(quant.FormatQ4_0); err != nil {
			return nil, err
		}
		if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
			return nil, err
		}
		mr := r.Core
		mr.Z, mr.Y, mr.X, mr.L = 0, 0, 0, 1
		_ = g.BindOp(0, 0, 0, 1, mr, r)
		ms := s.Core
		ms.Z, ms.Y, ms.X, ms.L = 0, 0, 0, 2
		_ = g.BindOp(0, 0, 0, 2, ms, s)
		return g, nil
	}
	a, err := mk(false)
	if err != nil {
		return err
	}
	b, err := mk(true)
	if err != nil {
		return err
	}
	clone, err := evolution.CloneGrid(a)
	if err != nil {
		rec("multi", "mixed", "mixed", "FAIL", err.Error())
		return err
	}
	cmp := dna.CompareNetworks(dna.ExtractDNA(a), dna.ExtractDNA(clone))
	if cmp.OverallOverlap < 0.999 {
		return fmt.Errorf("clone drift overlap=%v", cmp.OverallOverlap)
	}
	if _, err := evolution.SpliceDNA(a, b, evolution.DefaultSpliceConfig()); err != nil {
		rec("multi", "mixed", "mixed", "FAIL", err.Error())
		return err
	}
	rec("multi", "mixed", "mixed", "OK", "")
	return nil
}
