package seven

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/webgpu"
)

const (
	denseDim   = 64
	denseBatch = 2
	denseLR    = 1e-2
)

func makeDenseInit(in, out int) []float32 {
	w := make([]float32, out*in)
	n := out
	if in < n {
		n = in
	}
	for i := 0; i < n; i++ {
		w[i*in+i] = 1
	}
	for i := range w {
		w[i] += float32((i%7)-3) * 0.01
	}
	return w
}

func makeDenseXY(batch, in, out int) (x, y *core.Tensor[float32]) {
	x = core.NewTensor[float32](batch, in)
	y = core.NewTensor[float32](batch, out)
	for i := range x.Data {
		x.Data[i] = float32((i%5)+1) * 0.1
	}
	for i := range y.Data {
		y.Data[i] = float32((i%3)+1) * 0.05
	}
	return x, y
}

func buildDenseGrid(dim int, be core.Backend, multi bool, dt core.DType, format quant.Format) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	g.Exec.Backend = be
	g.Exec.MultiCore = multi
	g.Exec.TileSize = 32
	l, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, makeDenseInit(dim, dim))
	if err != nil {
		return nil, err
	}
	if dt != core.DTypeFloat32 {
		if err := l.Weights.SetDType(dt); err != nil {
			return nil, err
		}
		l.Core.DType = dt
	}
	if format != quant.FormatNone {
		if err := l.Weights.Pack(format); err != nil {
			return nil, err
		}
	}
	if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
		return nil, err
	}
	return g, nil
}

func captureFwdBwd(g *architecture.Grid, x *core.Tensor[float32]) (post, dW []float32, err error) {
	fwd, err := forward.Forward(g, x)
	if err != nil {
		return nil, nil, err
	}
	gy := core.NewTensor[float32](fwd.Output.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		return nil, nil, err
	}
	post = suites.CloneF32(fwd.Output.Data)
	if len(bwd.GradWs) > 0 && bwd.GradWs[0].DW != nil {
		dW = suites.CloneF32(bwd.GradWs[0].DW.Data)
	}
	return post, dW, nil
}

func denseRepeatDet() error {
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	x, _ := makeDenseXY(denseBatch, denseDim, denseDim)
	const reps = 3
	for _, be := range backends {
		if be == core.BackendSIMD && !simd.Enabled() {
			fmt.Printf("(simd skip) ")
			continue
		}
		if be == core.BackendWebGPU && !webgpu.Available() {
			fmt.Printf("(gpu skip) ")
			continue
		}
		g, err := buildDenseGrid(denseDim, be, false, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return err
		}
		base, _, err := captureFwdBwd(g, x)
		if err != nil {
			return fmt.Errorf("%s: %w", be, err)
		}
		var max float64
		for i := 0; i < reps; i++ {
			g2, err := buildDenseGrid(denseDim, be, false, core.DTypeFloat32, quant.FormatNone)
			if err != nil {
				return err
			}
			p, _, err := captureFwdBwd(g2, x)
			if err != nil {
				return err
			}
			if d := suites.MaxAbsDiff(base, p); d > max {
				max = d
			}
		}
		if err := suites.RequireDet(be.String()+" repeat", max, suites.DetTolFwd); err != nil {
			return err
		}
		fmt.Printf("(%s Δ=%.3g) ", be, max)
	}
	return nil
}

func denseSCMCFormatNone() error {
	x, _ := makeDenseXY(denseBatch, denseDim, denseDim)
	var fails []string
	okN := 0
	for _, dt := range core.AllDTypes {
		gSC, err := buildDenseGrid(denseDim, core.BackendCPUTiled, false, dt, quant.FormatNone)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: build SC: %v", dt, err))
			continue
		}
		gMC, err := buildDenseGrid(denseDim, core.BackendCPUTiled, true, dt, quant.FormatNone)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: build MC: %v", dt, err))
			continue
		}
		pSC, wSC, err := captureFwdBwd(gSC, x)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: SC: %v", dt, err))
			continue
		}
		pMC, wMC, err := captureFwdBwd(gMC, x)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: MC: %v", dt, err))
			continue
		}
		dP := suites.MaxAbsDiff(pSC, pMC)
		dW := suites.MaxAbsDiff(wSC, wMC)
		if err := suites.RequireDet(dt.String()+" fwd SC↔MC", dP, suites.DetTolFwd); err != nil {
			fails = append(fails, err.Error())
			continue
		}
		if err := suites.RequireDet(dt.String()+" bwd SC↔MC", dW, suites.DetTolBwd); err != nil {
			fails = append(fails, err.Error())
			continue
		}
		okN++
	}
	fmt.Printf("(FormatNone SC↔MC %d/%d) ", okN, len(core.AllDTypes))
	if len(fails) > 0 {
		return fmt.Errorf("%d fails: %s", len(fails), joins(fails, 6))
	}
	return nil
}

func denseSCMCQuants() error {
	x, _ := makeDenseXY(denseBatch, denseDim, denseDim)
	var fails []string
	okN, gapN := 0, 0
	for _, f := range quant.AllFormats {
		if f == quant.FormatAffinePacked && !suites.AffinePackable(denseDim, denseDim) {
			gapN++
			continue
		}
		gSC, err := buildDenseGrid(denseDim, core.BackendCPUTiled, false, core.DTypeFloat32, f)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: build SC: %v", f, err))
			continue
		}
		gMC, err := buildDenseGrid(denseDim, core.BackendCPUTiled, true, core.DTypeFloat32, f)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: build MC: %v", f, err))
			continue
		}
		pSC, wSC, err := captureFwdBwd(gSC, x)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: SC: %v", f, err))
			continue
		}
		pMC, wMC, err := captureFwdBwd(gMC, x)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: MC: %v", f, err))
			continue
		}
		dP := suites.MaxAbsDiff(pSC, pMC)
		dW := suites.MaxAbsDiff(wSC, wMC)
		tolP, tolW := suites.DetTolFwd, suites.DetTolBwd
		if f != quant.FormatNone {
			tolP, tolW = 1e-3, 5e-3 // packed paths looser
		}
		if err := suites.RequireDet(f.String()+" fwd", dP, tolP); err != nil {
			fails = append(fails, err.Error())
			continue
		}
		if err := suites.RequireDet(f.String()+" bwd", dW, tolW); err != nil {
			fails = append(fails, err.Error())
			continue
		}
		okN++
	}
	fmt.Printf("(quants SC↔MC ok=%d gap=%d) ", okN, gapN)
	if len(fails) > 0 {
		return fmt.Errorf("%d fails: %s", len(fails), joins(fails, 6))
	}
	return nil
}

func trainEpochs(g *architecture.Grid, x, y *core.Tensor[float32], epochs int, lr float64) (first, last float64, err error) {
	for e := 0; e < epochs; e++ {
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return 0, 0, err
		}
		loss, err := training.Step(fwd, y, lr)
		if err != nil {
			return 0, 0, err
		}
		if e == 0 {
			first = loss
		}
		last = loss
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return first, last, fmt.Errorf("non-finite loss at epoch %d: %v", e, loss)
		}
	}
	return first, last, nil
}

func denseTrainParity() error {
	x, y := makeDenseXY(denseBatch, denseDim, denseDim)
	const epochs = 8

	gSC, err := buildDenseGrid(denseDim, core.BackendCPUTiled, false, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	first, last, err := trainEpochs(gSC, x, y, epochs, denseLR)
	if err != nil {
		return fmt.Errorf("SC train: %w", err)
	}
	if !(last < first) {
		return fmt.Errorf("SC train did not reduce loss: first=%g last=%g", first, last)
	}
	fmt.Printf("(SC learn %.4g→%.4g) ", first, last)

	// Rebuild identical init for MC / SIMD parity of final weights after same epochs.
	runTrainWeights := func(be core.Backend, multi bool) ([]float32, float64, error) {
		g, err := buildDenseGrid(denseDim, be, multi, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return nil, 0, err
		}
		_, last, err := trainEpochs(g, x, y, epochs, denseLR)
		if err != nil {
			return nil, 0, err
		}
		op := g.Cells[0].Op.(*dense.Layer)
		w, ok := op.Weights.MasterF32()
		if !ok {
			return nil, 0, fmt.Errorf("no MasterF32")
		}
		return suites.CloneF32(w), last, nil
	}

	wSC, lossSC, err := runTrainWeights(core.BackendCPUTiled, false)
	if err != nil {
		return err
	}
	wMC, lossMC, err := runTrainWeights(core.BackendCPUTiled, true)
	if err != nil {
		return err
	}
	if d := suites.MaxAbsDiff(wSC, wMC); d > suites.DetTolTrain {
		return fmt.Errorf("train SC↔MC weight Δ=%g lossSC=%g lossMC=%g", d, lossSC, lossMC)
	}
	fmt.Printf("(SC↔MC wΔ=%.3g) ", suites.MaxAbsDiff(wSC, wMC))

	if simd.Enabled() {
		wSIMD, lossSIMD, err := runTrainWeights(core.BackendSIMD, false)
		if err != nil {
			return fmt.Errorf("SIMD train: %w", err)
		}
		if d := suites.MaxAbsDiff(wSC, wSIMD); d > suites.DetTolTrain {
			return fmt.Errorf("train SC↔SIMD weight Δ=%g lossSC=%g lossSIMD=%g", d, lossSC, lossSIMD)
		}
		fmt.Printf("(SC↔SIMD wΔ=%.3g) ", suites.MaxAbsDiff(wSC, wSIMD))
	}
	return nil
}

func denseEntityTrain() error {
	x, y := makeDenseXY(denseBatch, denseDim, denseDim)
	g, err := buildDenseGrid(denseDim, core.BackendCPUTiled, false, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "w2a-seven-entity-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	checkReload := func(phase string, refLoss float64) error {
		path := filepath.Join(dir, phase+".entity")
		if err := serialization.SaveEntity(path, g); err != nil {
			return fmt.Errorf("%s save: %w", phase, err)
		}
		g2, err := serialization.LoadEntity(path)
		if err != nil {
			return fmt.Errorf("%s load: %w", phase, err)
		}
		fwd1, err := forward.Forward(g, x)
		if err != nil {
			return err
		}
		fwd2, err := forward.Forward(g2, x)
		if err != nil {
			return err
		}
		if d := suites.MaxAbsDiff(fwd1.Output.Data, fwd2.Output.Data); d > suites.DetTolFwd {
			return fmt.Errorf("%s entity fwd Δ=%g", phase, d)
		}
		loss, err := training.MSE(fwd2.Output, y)
		if err != nil {
			return err
		}
		if math.Abs(loss-refLoss) > 1e-4 && refLoss > 0 {
			// allow small drift after train; before train should match closely
			if phase == "before" {
				return fmt.Errorf("%s entity loss %g vs ref %g", phase, loss, refLoss)
			}
		}
		return nil
	}

	fwd0, err := forward.Forward(g, x)
	if err != nil {
		return err
	}
	loss0, err := training.MSE(fwd0.Output, y)
	if err != nil {
		return err
	}
	if err := checkReload("before", loss0); err != nil {
		return err
	}
	_, last, err := trainEpochs(g, x, y, 6, denseLR)
	if err != nil {
		return err
	}
	if !(last < loss0) {
		return fmt.Errorf("train did not learn: %g → %g", loss0, last)
	}
	if err := checkReload("after", last); err != nil {
		return err
	}
	fmt.Printf("(entity before+after learn %.4g→%.4g) ", loss0, last)
	return nil
}

func denseShapeTiers() error {
	for _, tier := range suites.ShapeTier() {
		dim := tier.Dim
		if dim%64 != 0 && dim < 64 {
			// keep Affine-friendly when possible; S=32 still ok for FormatNone
		}
		x, y := makeDenseXY(2, dim, dim)
		g, err := buildDenseGrid(dim, core.BackendCPUTiled, false, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return fmt.Errorf("%s: %w", tier.Name, err)
		}
		if _, _, err := captureFwdBwd(g, x); err != nil {
			return fmt.Errorf("%s fwd/bwd: %w", tier.Name, err)
		}
		first, last, err := trainEpochs(g, x, y, 3, denseLR)
		if err != nil {
			return fmt.Errorf("%s train: %w", tier.Name, err)
		}
		if math.IsNaN(last) {
			return fmt.Errorf("%s non-finite loss", tier.Name)
		}
		fmt.Printf("(%s dim=%d %.4g→%.4g) ", tier.Name, dim, first, last)
	}
	return nil
}

func joins(ss []string, n int) string {
	if len(ss) <= n {
		return fmt.Sprintf("%v", ss)
	}
	return fmt.Sprintf("%v … (+%d)", ss[:n], len(ss)-n)
}
