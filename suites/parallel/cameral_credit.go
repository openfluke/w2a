package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

// CameralCreditModes measures Split / FastProxy / Sparse / Alt (scorecard §9)
// on the cameral graphs the timed test41 matrix does not cover.
func CameralCreditModes() error {
	var fails []string
	fmt.Printf("\n  CAMERAL CREDIT — Split/Alt family × sandwich graphs\n")

	for _, mode := range parallel.AllCreditTrainModes() {
		status, note := runCameralMode(mode)
		rec("cameral_credit_"+mode.String(), "float32", "none", "cpu_tiled", "bicameral", status, note)
		fmt.Printf("  %-28s %8s  %s\n", mode.String(), status, note)
		if status == "FAIL" {
			fails = append(fails, mode.String()+":"+note)
		}
	}

	for _, mode := range parallel.AllMeshCreditTrainModes() {
		status, note := runCameralMode(mode)
		rec("cameral_mesh_credit_"+mode.String(), "float32", "none", "cpu_tiled", "bicameral", status, note)
		fmt.Printf("  mesh %-23s %8s  %s\n", mode.String(), status, note)
		if status == "FAIL" {
			fails = append(fails, mode.String()+":"+note)
		}
	}

	for _, n := range []int{1, 3} {
		status, note := runCreditHemiN(n, parallel.ModeTweenSplitFastProxy)
		rec(fmt.Sprintf("cameral_credit_hemi%d_fastproxy", n), "float32", "none", "cpu_tiled", "sandwich", status, note)
		fmt.Printf("  hemi%-2d FastProxy                 %8s  %s\n", n, status, note)
		if status == "FAIL" {
			fails = append(fails, fmt.Sprintf("hemi%d:%s", n, note))
		}
	}

	for _, c := range []parallel.CombineMode{
		parallel.CombineAdd, parallel.CombineAvg, parallel.CombineConcat, parallel.CombineFilter,
	} {
		status, note := runCreditCombine(c)
		rec("cameral_credit_combine_"+string(c), "float32", "none", "cpu_tiled", "parallel", status, note)
		fmt.Printf("  combine %-8s                 %8s  %s\n", c, status, note)
		if status == "FAIL" {
			fails = append(fails, string(c)+":"+note)
		}
	}

	status, note := runCreditMixedFastProxyBP()
	rec("cameral_credit_mixed_fastproxy_bp", "float32", "none", "cpu_tiled", "sandwich", status, note)
	fmt.Printf("  mixed FastProxy∥StepBP          %8s  %s\n", status, note)
	if status == "FAIL" {
		fails = append(fails, "mixed:"+note)
	}

	okN, gapN := 0, 0
	for _, k := range polyKinds() {
		status, note := runCameralPolyKind(k, parallel.ModeTweenSplitFastProxy)
		rec("cameral_credit_poly_"+k.name+"_fastproxy", "float32", "none", "cpu_tiled", "stack", status, note)
		fmt.Printf("  poly %-16s FastProxy %8s  %s\n", k.name, status, note)
		switch status {
		case "OK":
			okN++
		case "GAP":
			gapN++
		case "FAIL":
			fails = append(fails, k.name+":"+note)
		}
	}
	fmt.Printf("  poly FastProxy: %d OK, %d GAP\n", okN, gapN)

	if len(fails) > 0 {
		n := min(8, len(fails))
		return fmt.Errorf("cameral credit: %s", strings.Join(fails[:n], "; "))
	}
	return nil
}

func runCreditHemiN(n int, mode parallel.TrainMode) (status, note string) {
	const dim = 12
	stem, err := dense.New(dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	hemi, err := parallel.Hemispheres(dim, dim, n, parallel.CombineAdd, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return "FAIL", err.Error()
	}
	head, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	s, err := parallel.Sandwich(stem, hemi, head)
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
	x := core.NewTensor[float32](2, dim)
	y := core.NewTensor[float32](2, dim)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.25
	}
	if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.1); err != nil {
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
	return "OK", fmt.Sprintf("n=%d Δelems=%d max|Δ|=%.6g", n, delta, maxAbs)
}

func runCreditCombine(c parallel.CombineMode) (status, note string) {
	cfg := parallel.Config{Dim: 12, OutFeat: 12, Branches: 2, Combine: c}
	l, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(l); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	before, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", err.Error()
	}
	x := core.NewTensor[float32](2, cfg.Dim)
	fillOnes(x)
	_, post, err := parallel.Forward(l, x)
	if err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.2
	}
	if _, err := parallel.TrainMSE(l, x, y, parallel.ModeTweenSplitFastProxy, 0.1); err != nil {
		return "FAIL", err.Error()
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "weights unchanged"
	}
	return "OK", fmt.Sprintf("%s Δelems=%d max|Δ|=%.6g", c, delta, maxAbs)
}

func runCreditMixedFastProxyBP() (status, note string) {
	s, err := parallel.Bicameral(10, 16, 4, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	hemi, ok := s.Children[1].(*parallel.Layer)
	if !ok {
		return "FAIL", "expected Parallel hemi"
	}
	hemi.SetBranchModes(parallel.ModeTweenSplitFastProxy, parallel.ModeStepBP)
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
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeStepBP, 0.1); err != nil {
		return "FAIL", err.Error()
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "mixed hemi weights unchanged"
	}
	return "OK", fmt.Sprintf("FastProxy∥StepBP Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}
