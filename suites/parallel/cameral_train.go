package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/dna"
)

// CameralTrainModes: Bicameral trains under all 9 test41 modes, mixed
// per-hemisphere modes, and grid Step / StepTween / StepMesh paths.
func CameralTrainModes() error {
	var fails []string
	for _, mode := range parallel.AllConcreteTrainModes() {
		status, note := runCameralMode(mode)
		rec("cameral_train_"+mode.String(), "float32", "none", "cpu_tiled", "1x1x1x1", status, note)
		if status != "OK" && status != "GAP" {
			fails = append(fails, mode.String()+":"+note)
		}
	}
	status, note := runCameralMixedHemispheres()
	rec("cameral_train_mixed_hemi", "float32", "none", "cpu_tiled", "1x1x1x1", status, note)
	if status != "OK" {
		fails = append(fails, "mixed:"+note)
	}
	status, note = runCameralGridAllFamilies()
	rec("cameral_train_grid_all", "float32", "none", "cpu_tiled", "1x1x1x1", status, note)
	if status != "OK" {
		fails = append(fails, "grid:"+note)
	}
	if len(fails) > 0 {
		return fmt.Errorf("cameral train modes: %s", strings.Join(fails, "; "))
	}
	return nil
}

func runCameralMode(mode parallel.TrainMode) (status, note string) {
	s, err := parallel.Bicameral(10, 16, 4, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	before, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", err.Error()
	}
	x := core.NewTensor[float32](2, 10)
	y := core.NewTensor[float32](2, 4)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.25
	}
	if err := trainStackByMode(s, x, y, mode, 0.1); err != nil {
		return "FAIL", err.Error()
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "weights unchanged"
	}
	return "OK", fmt.Sprintf("%s Δelems=%d max|Δ|=%.6g", mode, delta, maxAbs)
}

// trainStackByMode dispatches Stack-local TrainStackMSE for seq modes, or a
// 1×1×1×1 grid Step/StepTween/StepMesh for Mesh* (and exercises Step* via grid too).
func trainStackByMode(s *parallel.Stack, x, y *core.Tensor[float32], mode parallel.TrainMode, lr float64) error {
	if s == nil {
		return fmt.Errorf("nil stack")
	}
	switch {
	case mode.RequiresGrid():
		return trainStackOnGrid(s, x, y, mode, lr)
	case mode == parallel.ModeStepBP || mode == parallel.ModeStepTween || mode == parallel.ModeStepTweenChain:
		// Online Step* — still one TrainStackMSE call (batch already small); also poke grid once.
		if _, err := parallel.TrainStackMSE(s, x, y, mode, lr); err != nil {
			return err
		}
		return trainStackOnGrid(s, x, y, mode, lr)
	default:
		_, err := parallel.TrainStackMSE(s, x, y, mode, lr)
		return err
	}
}

func trainStackOnGrid(s *parallel.Stack, x, y *core.Tensor[float32], mode parallel.TrainMode, lr float64) error {
	g := architecture.NewGrid(1, 1, 1, 1)
	if err := parallel.PlaceStack(g, 0, 0, 0, 0, s); err != nil {
		return err
	}
	switch mode {
	case parallel.ModeMeshBP, parallel.ModeStepBP, parallel.ModeNormalBP:
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return err
		}
		_, err = training.Step(fwd, y, lr)
		return err
	case parallel.ModeMeshTween, parallel.ModeStepTween, parallel.ModeTween:
		return gridTween(g, x, y, lr, false)
	case parallel.ModeMeshTweenChain, parallel.ModeStepTweenChain, parallel.ModeTweenChain:
		return gridTween(g, x, y, lr, true)
	default:
		_, err := parallel.TrainStackMSE(s, x, y, mode, lr)
		return err
	}
}

func gridTween(g *architecture.Grid, x, y *core.Tensor[float32], lr float64, chain bool) error {
	if chain {
		_, _, err := training.StepTween(g, x, y, lr) // ApplyTween defaults UseChainRule=true
		return err
	}
	// Non-chain: mesh-style gap tween via step ticks + ApplyTween (UseChainRule=false).
	_, _, err := training.StepMesh(g, x, y, 1, lr)
	return err
}

func runCameralMixedHemispheres() (status, note string) {
	const dim = 12
	// Stamp first 9 modes onto as many hemispheres (here: 2 for smoke — SGD∥Tween family).
	left, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	right, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	hemi, err := parallel.HemispheresFrom(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
	}, []any{left, right}, nil)
	if err != nil {
		return "FAIL", err.Error()
	}
	hemi.SetBranchModes(parallel.ModeNormalBP, parallel.ModeTween)
	stem, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	head, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	root, err := parallel.Sandwich(stem, hemi, head)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(root); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	before, err := dna.FlattenOp(root)
	if err != nil {
		return "FAIL", err.Error()
	}
	x := core.NewTensor[float32](2, dim)
	y := core.NewTensor[float32](2, dim)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.3
	}
	if _, err := parallel.TrainStackMSE(root, x, y, parallel.ModeNormalBP, 0.1); err != nil {
		return "FAIL", err.Error()
	}
	after, err := dna.FlattenOp(root)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "mixed hemi weights unchanged"
	}
	return "OK", fmt.Sprintf("NormalBP∥Tween Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}

func runCameralGridAllFamilies() (status, note string) {
	s, err := parallel.Bicameral(8, 12, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	x := core.NewTensor[float32](1, 8)
	y := core.NewTensor[float32](1, 4)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.2
	}
	before, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", err.Error()
	}
	// Exercise BP + tween + mesh families on the same sandwich.
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepBP,
		parallel.ModeStepTween,
		parallel.ModeStepTweenChain,
		parallel.ModeMeshBP,
		parallel.ModeMeshTween,
		parallel.ModeMeshTweenChain,
	} {
		if err := trainStackOnGrid(s, x, y, mode, 0.1); err != nil {
			return "FAIL", mode.String()+": "+err.Error()
		}
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "grid Step/Mesh family weights unchanged"
	}
	return "OK", fmt.Sprintf("Step+Mesh families Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}
