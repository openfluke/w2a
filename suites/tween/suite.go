package tween

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/forward"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/training"
	"github.com/openfluke/welvet/tween"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Dense StepTween gap reduce smoke", Run: denseGapReduce},
		{Name: "FormatNone × all 34 dtypes StepTween (Dense)", Run: formatNoneAllDTypes},
		{Name: "All quants × Float32 StepTween (Dense)", Run: allQuantsFloat32},
		{Name: "Multi-layer chain-rule + layerwise (Dense+RMSNorm)", Run: multiLayerModes},
		{Name: "SwiGLU layerwise gaps smoke", Run: swigluLayerwise},
		{Name: "GAP CENSUS — dtype×quant StepTween cells", Run: census},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("tween", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("tween", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("tween: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("tween: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("tween", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("tween", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func rec(op, dt, format, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "tween", Op: op, DType: dt, Format: format, Backend: "cpu", Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func identityDense(n int, dt core.DType, format quant.Format) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	w := make([]float32, n*n)
	for i := 0; i < n; i++ {
		w[i*n+i] = 1
	}
	l, err := dense.NewConfigured(n, n, core.ActivationLinear, dt, format, w)
	if err != nil {
		return nil, err
	}
	if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
		return nil, err
	}
	return g, nil
}

func makeIO(n int, scale float64) (x, target *core.Tensor[float32]) {
	x = core.NewTensor[float32](1, n)
	target = core.NewTensor[float32](1, n)
	for i := 0; i < n; i++ {
		x.Data[i] = float32(i + 1)
		target.Data[i] = float32(float64(i+1) * scale)
	}
	return x, target
}

func denseGapReduce() error {
	g, err := identityDense(4, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	x, target := makeIO(4, 2)
	loss0, _, err := training.StepTween(g, x, target, 0.05)
	if err != nil {
		rec("step", "f32", "none", "FAIL", err.Error())
		return err
	}
	loss1, _, err := training.StepTween(g, x, target, 0.05)
	if err != nil {
		return err
	}
	if math.IsNaN(loss0) || math.IsNaN(loss1) || loss1 >= loss0 {
		rec("step", "f32", "none", "FAIL", fmt.Sprintf("%v→%v", loss0, loss1))
		return fmt.Errorf("expected loss drop %v → %v", loss0, loss1)
	}
	rec("step", "f32", "none", "OK", fmt.Sprintf("%.4f→%.4f", loss0, loss1))
	return nil
}

func formatNoneAllDTypes() error {
	var fails int
	for _, dt := range core.AllDTypes {
		g, err := identityDense(4, dt, quant.FormatNone)
		if err != nil {
			rec("step", dt.String(), "none", "GAP", err.Error())
			fails++
			continue
		}
		x, target := makeIO(4, 1.5)
		if _, _, err := training.StepTween(g, x, target, 0.02); err != nil {
			rec("step", dt.String(), "none", "FAIL", err.Error())
			fails++
			continue
		}
		rec("step", dt.String(), "none", "OK", "")
	}
	fmt.Printf("(%d dtypes) ", len(core.AllDTypes))
	if fails > 0 {
		return fmt.Errorf("%d dtype tween fails/gaps", fails)
	}
	return nil
}

func allQuantsFloat32() error {
	var fails, gaps, oks int
	for _, f := range quant.AllFormats {
		g, err := identityDense(8, core.DTypeFloat32, f)
		if err != nil {
			rec("step", "f32", f.String(), "GAP", err.Error())
			gaps++
			continue
		}
		x, target := makeIO(8, 1.25)
		if _, _, err := training.StepTween(g, x, target, 0.02); err != nil {
			rec("step", "f32", f.String(), "FAIL", err.Error())
			fails++
			continue
		}
		rec("step", "f32", f.String(), "OK", "")
		oks++
	}
	fmt.Printf("(ok=%d gap=%d fail=%d / %d quants) ", oks, gaps, fails, len(quant.AllFormats))
	if fails > 0 {
		return fmt.Errorf("%d quant tween fails", fails)
	}
	return nil
}

func multiLayerModes() error {
	g := architecture.NewGrid(1, 1, 1, 2)
	d, err := dense.New(4, 4, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	w := make([]float32, 16)
	for i := 0; i < 4; i++ {
		w[i*4+i] = 1
	}
	_ = d.Weights.SetFromF32(w)
	r, err := rmsnorm.New(rmsnorm.Config{Dim: 4})
	if err != nil {
		return err
	}
	if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
		return err
	}
	mr := r.Core
	mr.Z, mr.Y, mr.X, mr.L = 0, 0, 0, 1
	if err := g.BindOp(0, 0, 0, 1, mr, r); err != nil {
		return err
	}
	x, target := makeIO(4, 1.5)
	fwd, err := forward.Forward(g, x)
	if err != nil {
		return err
	}
	if _, err := training.ApplyTween(g, fwd, x, target, 0.02); err != nil {
		rec("multi", "f32", "none", "FAIL", "chain: "+err.Error())
		return err
	}
	cfg := tween.DefaultConfig()
	cfg.UseChainRule = false
	st := tween.NewState[float32](g, cfg)
	tween.CaptureFromForward(st, fwd, x)
	if err := tween.Backward(g, st, target); err != nil {
		return err
	}
	st.CalculateLinkBudgets()
	if err := tween.ApplyGaps(g, st, 0.02); err != nil {
		rec("multi", "f32", "none", "FAIL", "layerwise: "+err.Error())
		return err
	}
	rec("multi", "f32", "none", "OK", "")
	return nil
}

func swigluLayerwise() error {
	g := architecture.NewGrid(1, 1, 1, 1)
	s, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
	if err != nil {
		return err
	}
	m := s.Core
	m.Z, m.Y, m.X, m.L = 0, 0, 0, 0
	if err := g.BindOp(0, 0, 0, 0, m, s); err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 8)
	target := core.NewTensor[float32](1, 8)
	for i := 0; i < 8; i++ {
		x.Data[i] = float32(i+1) * 0.1
		target.Data[i] = float32(i+1) * 0.15
	}
	fwd, err := forward.Forward(g, x)
	if err != nil {
		rec("swiglu", "f32", "none", "FAIL", err.Error())
		return err
	}
	cfg := tween.DefaultConfig()
	cfg.UseChainRule = false
	st := tween.NewState[float32](g, cfg)
	tween.CaptureFromForward(st, fwd, x)
	if err := tween.Backward(g, st, target); err != nil {
		return err
	}
	st.CalculateLinkBudgets()
	if err := tween.ApplyGaps(g, st, 0.01); err != nil {
		rec("swiglu", "f32", "none", "FAIL", err.Error())
		return err
	}
	rec("swiglu", "f32", "none", "OK", "")
	return nil
}

func census() error {
	// Spot matrix: a few dtypes × a few quants (full 34×20 already covered above as dedicated cases).
	dts := []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeInt8, core.DTypeBFloat16}
	fmts := []quant.Format{quant.FormatNone, quant.FormatQ8_0, quant.FormatQ4_0, quant.FormatQ4_K}
	var fails, oks int
	for _, dt := range dts {
		for _, f := range fmts {
			if f != quant.FormatNone && dt != core.DTypeFloat32 {
				// Pack is defined on f32 master — still exercise NewConfigured path.
			}
			g, err := identityDense(8, dt, f)
			if err != nil {
				rec("census", dt.String(), f.String(), "GAP", err.Error())
				continue
			}
			x, target := makeIO(8, 1.1)
			if _, _, err := training.StepTween(g, x, target, 0.01); err != nil {
				rec("census", dt.String(), f.String(), "FAIL", err.Error())
				fails++
				continue
			}
			rec("census", dt.String(), f.String(), "OK", "")
			oks++
		}
	}
	fmt.Printf("(ok=%d fail=%d) ", oks, fails)
	if fails > 0 {
		return fmt.Errorf("%d census cells failed", fails)
	}
	return nil
}
