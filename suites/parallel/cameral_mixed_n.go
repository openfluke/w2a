package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/systems/dna"
)

// distinctHemiModes returns n train modes cycling the full test41 set
// (NormalBP…MeshTweenChain). For n=9 each hemisphere gets a unique mode.
func distinctHemiModes(n int) []parallel.TrainMode {
	base := parallel.AllConcreteTrainModes()
	out := make([]parallel.TrainMode, n)
	for i := 0; i < n; i++ {
		out[i] = base[i%len(base)]
	}
	return out
}

func modesLabel(modes []parallel.TrainMode) string {
	parts := make([]string, len(modes))
	for i, m := range modes {
		parts[i] = m.String()
	}
	return strings.Join(parts, "∥")
}

// makePolyN expands a polyKind maker to n hemisphere Ops (re-calling make as needed).
func makePolyN(k polyKind, n int) (branches []any, cfg parallel.Config, x *core.Tensor[float32], trainable bool, err error) {
	if n < 1 {
		return nil, parallel.Config{}, nil, false, fmt.Errorf("makePolyN: n≥1")
	}
	var all []any
	for len(all) < n {
		br, c, xin, t, e := k.make()
		if e != nil {
			return nil, parallel.Config{}, nil, false, e
		}
		cfg, x, trainable = c, xin, t
		all = append(all, br...)
	}
	cfg.Branches = n
	return all[:n], cfg, x, trainable, nil
}

func setStackHemiModes(s *parallel.Stack, modes ...parallel.TrainMode) error {
	if s == nil {
		return fmt.Errorf("nil stack")
	}
	for _, ch := range s.Children {
		switch v := ch.(type) {
		case *parallel.Layer:
			if len(modes) > 0 {
				v.SetBranchModes(modes...)
			}
			return nil
		case *parallel.Stack:
			if err := setStackHemiModes(v, modes...); err == nil {
				return nil
			}
		}
	}
	return fmt.Errorf("no Parallel hemisphere in stack")
}

// CameralMixedHemiNSmoke: Dense hemi3 (first 3 test41 modes) and hemi9
// (all 9 distinct modes) sandwiches — weight-delta under TrainStackMSE.
func CameralMixedHemiNSmoke() error {
	var fails []string
	for _, n := range []int{3, 9} {
		status, note := runDenseMixedHemiN(n, core.DTypeFloat32)
		rec(fmt.Sprintf("cameral_mixed_hemi%d", n), "float32", "none", "cpu_tiled", "sandwich", status, note)
		fmt.Printf("  hemi%d %-60s %8s  %s\n", n, modesLabel(distinctHemiModes(n)), status, note)
		if status == "FAIL" {
			fails = append(fails, fmt.Sprintf("hemi%d: %s", n, note))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("cameral mixed hemi-n: %s", strings.Join(fails, "; "))
	}
	return nil
}

func runDenseMixedHemiN(n int, dt core.DType) (status, note string) {
	const dim = 12
	modes := distinctHemiModes(n)
	branches := make([]any, n)
	for i := 0; i < n; i++ {
		ch, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return "FAIL", err.Error()
		}
		branches[i] = ch
	}
	hemi, err := parallel.HemispheresFrom(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: n, Combine: parallel.CombineAdd,
	}, branches, nil)
	if err != nil {
		return "FAIL", err.Error()
	}
	hemi.SetBranchModes(modes...)
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
	if dt != core.DTypeFloat32 {
		if err := root.SetDType(dt); err != nil {
			return "GAP", "SetDType: " + err.Error()
		}
	}
	if err := seedNonZero(root); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	x := core.NewTensor[float32](2, dim)
	y := core.NewTensor[float32](2, dim)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.3
	}
	before, err := dna.FlattenOp(root)
	if err != nil {
		return "FAIL", err.Error()
	}
	if _, err := parallel.TrainStackMSE(root, x, y, parallel.ModeNormalBP, 0.15); err != nil {
		return "FAIL", "train: " + err.Error()
	}
	after, err := dna.FlattenOp(root)
	if err != nil {
		return "FAIL", err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "GAP", "weights unchanged under " + modesLabel(modes) + "/" + dt.String()
	}
	return "OK", fmt.Sprintf("%s Δelems=%d max|Δ|=%.6g", modesLabel(modes), delta, maxAbs)
}

// CameralDenseMixedModesAllDTypes: Dense hemi9 with all distinct test41 modes × all 34 dtypes.
func CameralDenseMixedModesAllDTypes() error {
	const n = 9
	modes := distinctHemiModes(n)
	fmt.Printf("\n  CAMERAL MIXED — Dense hemi%d %s × all dtypes\n", n, modesLabel(modes))
	var fails []string
	var okN, gapN int
	for _, dt := range core.AllDTypes {
		status, note := runDenseMixedHemiN(n, dt)
		rec("cameral_mixed_hemi9_dense", dt.String(), "None", "cpu_tiled", "sandwich", status, note)
		fmt.Printf("  %-12s %8s  %s\n", dt.String(), status, note)
		switch status {
		case "OK":
			okN++
		case "GAP":
			gapN++
		default:
			fails = append(fails, fmt.Sprintf("%s:%s", dt, note))
		}
	}
	fmt.Printf("  summary: %d OK, %d GAP, %d FAIL (of %d)\n", okN, gapN, len(fails), len(core.AllDTypes))
	if len(fails) > 0 {
		nShow := min(10, len(fails))
		return fmt.Errorf("dense mixed hemi9×dtypes: %d failed (first): %s", len(fails), strings.Join(fails[:nShow], "; "))
	}
	return nil
}

// CameralPolyMixedModesAllKinds: every poly kind as hemi3 + hemi9 with distinct
// per-hemi train modes (float32).
func CameralPolyMixedModesAllKinds() error {
	kinds := polyKinds()
	ns := []int{3, 9}
	fmt.Printf("\n  CAMERAL POLY MIXED — kinds × hemi{3,9} distinct test41 modes (float32)\n")
	fmt.Printf("  kinds=%d ns=%v cells=%d\n\n", len(kinds), ns, len(kinds)*len(ns))

	var fails []string
	var okN, gapN int
	for _, k := range kinds {
		for _, n := range ns {
			status, note := runCameralPolyMixedN(k, n, core.DTypeFloat32)
			rec(fmt.Sprintf("cameral_poly_mixed_hemi%d_%s", n, k.name), "float32", "none", "cpu_tiled", "stack", status, note)
			fmt.Printf("  %-16s hemi%d %8s  %s\n", k.name, n, status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			default:
				fails = append(fails, fmt.Sprintf("%s/hemi%d:%s", k.name, n, note))
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL\n", okN, gapN, len(fails))
	if len(fails) > 0 {
		nShow := min(10, len(fails))
		return fmt.Errorf("cameral poly mixed kinds: %s", strings.Join(fails[:nShow], "; "))
	}
	return nil
}

// CameralPolyMixedModesAllKindsDTypes: hemi9 distinct modes × all layer kinds × all 34 dtypes.
func CameralPolyMixedModesAllKindsDTypes() error {
	const n = 9
	kinds := polyKinds()
	modes := distinctHemiModes(n)
	fmt.Printf("\n  CAMERAL POLY MIXED — kinds × hemi%d distinct modes × all dtypes\n", n)
	fmt.Printf("  modes=%s\n", modesLabel(modes))
	fmt.Printf("  kinds=%d dtypes=%d cells=%d\n\n", len(kinds), len(core.AllDTypes), len(kinds)*len(core.AllDTypes))

	var fails []string
	var okN, gapN int
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			status, note := runCameralPolyMixedN(k, n, dt)
			rec(fmt.Sprintf("cameral_poly_mixed_hemi%d_%s", n, k.name), dt.String(), "None", "cpu_tiled", "stack", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			default:
				fails = append(fails, fmt.Sprintf("%s/%s:%s", k.name, dt, note))
			}
		}
		fmt.Printf("  %-16s done\n", k.name)
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d)\n", okN, gapN, len(fails), len(kinds)*len(core.AllDTypes))
	if len(fails) > 0 {
		nShow := min(12, len(fails))
		return fmt.Errorf("cameral poly mixed kinds×dtypes: %d failed (first): %s", len(fails), strings.Join(fails[:nShow], "; "))
	}
	return nil
}

func runCameralPolyMixedN(k polyKind, n int, dt core.DType) (status, note string) {
	modes := distinctHemiModes(n)
	branches, cfg, x, trainable, err := makePolyN(k, n)
	if err != nil {
		return "FAIL", "make: " + err.Error()
	}
	s, err := parallel.CameralFromBranches(cfg, branches, nil)
	if err != nil {
		return "FAIL", "cameral: " + err.Error()
	}
	if err := setStackHemiModes(s, modes...); err != nil {
		return "FAIL", err.Error()
	}
	s.Exec.Backend = core.BackendCPUTiled
	s.SyncChildExec()
	if dt != core.DTypeFloat32 {
		if err := s.SetDType(dt); err != nil {
			return "GAP", "SetDType: " + err.Error()
		}
	}
	if k.name != "embedding" {
		fillOnes(x)
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}

	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.2
	}

	before, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", "snapshot: " + err.Error()
	}
	// Parent NormalBP; BranchModes override per hemisphere (Step/Mesh collapse to Family).
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeNormalBP, 0.15); err != nil {
		return "FAIL", "train: " + err.Error()
	}
	if !trainable {
		return "OK", "fwd+train (no trainable stores)"
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		return "FAIL", "snapshot after: " + err.Error()
	}
	if len(before) == 0 || len(after) == 0 {
		return "FAIL", "empty FlattenOp weights"
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		// Coarse dtypes / soft tween gaps — same policy as cameral poly.
		return "GAP", fmt.Sprintf("weights unchanged under %s/%s", modesLabel(modes), dt.String())
	}
	return "OK", fmt.Sprintf("Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}
