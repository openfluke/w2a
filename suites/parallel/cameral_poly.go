package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/systems/dna"
)

// CameralPolyAllKinds: every major Op kind as dual hemispheres inside a cameral
// Stack, trained under all 9 test41 train modes (plus Dense∥SwiGLU mixed).
func CameralPolyAllKinds() error {
	modes := parallel.AllConcreteTrainModes()
	var fails []string
	var okN, gapN int

	fmt.Printf("\n  CAMERAL POLY — all layer kinds × all train modes (Stack[Parallel∥Parallel])\n")
	kinds := polyKinds()
	fmt.Printf("  kinds=%d modes=%d cells=%d\n\n", len(kinds), len(modes), len(kinds)*len(modes)+len(modes))

	for _, k := range kinds {
		for _, mode := range modes {
			status, note := runCameralPolyKind(k, mode)
			rec("cameral_poly_"+k.name+"_"+mode.String(), "float32", "none", "cpu_tiled", "stack", status, note)
			fmt.Printf("  %-16s %-14s %8s  %s\n", k.name, mode.String(), status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			case "FAIL":
				fails = append(fails, fmt.Sprintf("%s/%s: %s", k.name, mode, note))
			}
		}
	}
	for _, mode := range modes {
		status, note := runCameralMixedDenseSwiGLU(mode)
		rec("cameral_poly_mixed_dense_swiglu_"+mode.String(), "float32", "none", "cpu_tiled", "stack", status, note)
		fmt.Printf("  %-16s %-14s %8s  %s\n", "mixed_dense∥swiglu", mode.String(), status, note)
		switch status {
		case "OK":
			okN++
		case "GAP":
			gapN++
		case "FAIL":
			fails = append(fails, fmt.Sprintf("mixed/%s: %s", mode, note))
		}
	}

	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL\n", okN, gapN, len(fails))
	if len(fails) > 0 {
		n := min(8, len(fails))
		return fmt.Errorf("cameral poly kinds: %s", strings.Join(fails[:n], "; "))
	}
	return nil
}

func runCameralPolyKind(k polyKind, mode parallel.TrainMode) (status, note string) {
	branches, cfg, x, trainable, err := k.make()
	if err != nil {
		return "FAIL", "make: " + err.Error()
	}
	s, err := parallel.CameralFromBranches(cfg, branches, nil)
	if err != nil {
		return "FAIL", "cameral: " + err.Error()
	}
	s.Exec.Backend = core.BackendCPUTiled
	s.SyncChildExec()
	if k.name != "embedding" {
		fillOnes(x)
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}

	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		return "FAIL", "fwd probe: " + err.Error()
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.2
	}

	before, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", "snapshot: " + err.Error()
	}
	if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.1); err != nil {
		return "FAIL", "train: " + err.Error()
	}
	if !trainable {
		return "OK", "fwd+train (no trainable stores)"
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", "snapshot after: " + err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "GAP", fmt.Sprintf("weights unchanged under %s (kind may absorb soft gaps)", mode)
	}
	return "OK", fmt.Sprintf("Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}

func runCameralMixedDenseSwiGLU(mode parallel.TrainMode) (status, note string) {
	const dim, batch = 32, 2
	left, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return "FAIL", err.Error()
	}
	right, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		return "FAIL", err.Error()
	}
	cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
	s, err := parallel.CameralFromBranches(cfg, []any{left, right}, nil)
	if err != nil {
		return "FAIL", err.Error()
	}
	s.Exec.Backend = core.BackendCPUTiled
	s.SyncChildExec()
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.25
	}
	before, _ := dna.FlattenOp(s)
	if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.1); err != nil {
		return "FAIL", err.Error()
	}
	after, _ := dna.FlattenOp(s)
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "GAP", "mixed Dense∥SwiGLU weights unchanged"
	}
	return "OK", fmt.Sprintf("Dense∥SwiGLU Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}
