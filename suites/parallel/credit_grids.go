package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/systems/dna"
)

// CameralCreditTrainGrids trains Split/FastProxy/Sparse/Alt and Mesh* credit
// on volumetric 1×1×1 / 2×2×2 / 3×3×3 cubes (CPU Float32 FormatNone).
// TimedTrainGrids* still uses training.Step (SGD) only.
func CameralCreditTrainGrids() error {
	var fails []string
	modes := append([]parallel.TrainMode(nil), parallel.AllCreditTrainModes()...)
	modes = append(modes, parallel.AllMeshCreditTrainModes()...)
	fmt.Printf("\n  CAMERAL CREDIT GRIDS — %d modes × 1³/2³/3³ (hop + per-cell credit)\n", len(modes))
	for _, n := range []int{1, 2, 3} {
		for _, mode := range modes {
			status, note := runCreditParallelCube(n, mode)
			grid := fmt.Sprintf("%dx%dx%d", n, n, n)
			rec("cameral_credit_cube_"+mode.String(), "float32", "none", "cpu_tiled", grid, status, note)
			fmt.Printf("  parallel %-4s %-28s %8s  %s\n", grid, mode.String(), status, note)
			if status == "FAIL" {
				fails = append(fails, fmt.Sprintf("P%s/%s:%s", grid, mode, note))
			}
		}
		for _, mode := range parallel.AllMeshCreditTrainModes() {
			status, note := runCreditStackCube(n, mode)
			grid := fmt.Sprintf("%dx%dx%d", n, n, n)
			rec("cameral_mesh_stack_cube_"+mode.String(), "float32", "none", "cpu_tiled", grid, status, note)
			fmt.Printf("  stack    %-4s %-28s %8s  %s\n", grid, mode.String(), status, note)
			if status == "FAIL" {
				fails = append(fails, fmt.Sprintf("S%s/%s:%s", grid, mode, note))
			}
		}
	}
	if len(fails) > 0 {
		k := min(8, len(fails))
		return fmt.Errorf("credit grids: %s", strings.Join(fails[:k], "; "))
	}
	return nil
}

func runCreditParallelCube(n int, mode parallel.TrainMode) (status, note string) {
	const dim = 8
	g := smokeGrid(n)
	cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
	l, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(l); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	if err := parallel.Place(g, 0, 0, 0, 0, l); err != nil {
		return "FAIL", "place: " + err.Error()
	}
	enableOrigin(g)
	return trainCreditCube(g, dim, dim, mode, 1)
}

func runCreditStackCube(n int, mode parallel.TrainMode) (status, note string) {
	const dim = 8
	g := smokeGrid(n)
	s, err := parallel.Bicameral(dim, dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := seedNonZero(s); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	if err := parallel.PlaceStack(g, 0, 0, 0, 0, s); err != nil {
		return "FAIL", "place: " + err.Error()
	}
	enableOrigin(g)
	return trainCreditCube(g, dim, dim, mode, 1)
}

func trainCreditCube(g *architecture.Grid, inDim, outDim int, mode parallel.TrainMode, cells int) (status, note string) {
	before, err := flattenGrid(g)
	if err != nil {
		return "FAIL", err.Error()
	}
	x := core.NewTensor[float32](2, inDim)
	y := core.NewTensor[float32](2, outDim)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.25
	}
	if _, err := forward.Forward(g, x); err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	if err := trainCreditCells(g, x, y, mode, 0.05); err != nil {
		return "FAIL", err.Error()
	}
	after, err := flattenGrid(g)
	if err != nil {
		return "FAIL", err.Error()
	}
	if hasNonFinite(after) {
		return "FAIL", "NaN/Inf weights after credit step"
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "weights unchanged"
	}
	return "OK", fmt.Sprintf("cells=%d Δelems=%d max|Δ|=%.6g", cells, delta, maxAbs)
}

func trainCreditCells(g *architecture.Grid, x, y *core.Tensor[float32], mode parallel.TrainMode, lr float64) error {
	for i := range g.Cells {
		if g.Cells[i].Layer.IsDisabled {
			continue
		}
		op := g.Cells[i].Op
		if op == nil {
			continue
		}
		switch v := op.(type) {
		case *parallel.Layer:
			if _, err := parallel.TrainMSE(v, x, y, mode, lr); err != nil {
				return fmt.Errorf("cell %d %s: %w", i, mode, err)
			}
		case *parallel.Stack:
			if _, err := parallel.TrainStackMSE(v, x, y, mode, lr); err != nil {
				return fmt.Errorf("cell %d %s: %w", i, mode, err)
			}
		default:
			return fmt.Errorf("cell %d unexpected op %T", i, op)
		}
	}
	return nil
}

func hasNonFinite(w []float32) bool {
	for _, v := range w {
		if v != v || v > 1e20 || v < -1e20 {
			return true
		}
	}
	return false
}

func flattenGrid(g *architecture.Grid) ([]float32, error) {
	if g == nil {
		return nil, fmt.Errorf("nil grid")
	}
	var flat []float32
	for i := range g.Cells {
		if g.Cells[i].Op == nil {
			continue
		}
		part, err := dna.FlattenOp(g.Cells[i].Op)
		if err != nil {
			return nil, err
		}
		flat = append(flat, part...)
	}
	return flat, nil
}

func buildPolyCube(k polyKind, n int) (*architecture.Grid, *core.Tensor[float32], bool, error) {
	g := smokeGrid(n)
	branches, cfg, x, trainable, err := k.make()
	if err != nil {
		return nil, nil, false, err
	}
	s, err := parallel.CameralFromBranches(cfg, branches, nil)
	if err != nil {
		return nil, nil, false, err
	}
	if err := seedNonZero(s); err != nil {
		return nil, nil, false, err
	}
	if err := parallel.PlaceStack(g, 0, 0, 0, 0, s); err != nil {
		return nil, nil, false, err
	}
	enableOrigin(g)
	return g, x, trainable, nil
}

func smokeGrid(n int) *architecture.Grid {
	g := architecture.NewGrid(n, n, n, 1)
	for i := range g.Cells {
		g.Cells[i].Layer.IsDisabled = true
	}
	return g
}

func enableOrigin(g *architecture.Grid) {
	if c := g.At(0, 0, 0, 0); c != nil {
		c.Layer.IsDisabled = false
	}
}

func seedGrid(g *architecture.Grid) error {
	for i := range g.Cells {
		if g.Cells[i].Op == nil {
			continue
		}
		if err := seedNonZero(g.Cells[i].Op); err != nil {
			return err
		}
	}
	return nil
}

func hopGrid(g *architecture.Grid, x *core.Tensor[float32]) error {
	_, err := forward.Forward(g, x)
	return err
}

func probeStackY(g *architecture.Grid, x *core.Tensor[float32]) (*core.Tensor[float32], error) {
	if g == nil || len(g.Cells) == 0 || g.Cells[0].Op == nil {
		return nil, fmt.Errorf("empty grid")
	}
	s, ok := g.Cells[0].Op.(*parallel.Stack)
	if !ok {
		return nil, fmt.Errorf("cell0 %T want *parallel.Stack", g.Cells[0].Op)
	}
	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		return nil, err
	}
	y := core.NewTensor[float32](post.Shape...)
	for i := range y.Data {
		y.Data[i] = 0.2
	}
	return y, nil
}

func trainPlacedCube(g *architecture.Grid, x, y *core.Tensor[float32], mode parallel.TrainMode, cells int, trainable bool, hopErr error) (status, note string) {
	before, err := flattenGrid(g)
	if err != nil {
		return "FAIL", err.Error()
	}
	if err := trainCreditCells(g, x, y, mode, 0.05); err != nil {
		return "FAIL", err.Error()
	}
	after, err := flattenGrid(g)
	if err != nil {
		return "FAIL", err.Error()
	}
	if hasNonFinite(after) {
		return "FAIL", "NaN/Inf weights"
	}
	hop := "hop"
	if hopErr != nil {
		hop = "hop GAP"
	}
	if !trainable {
		if hopErr != nil {
			return "GAP", hop + ": " + hopErr.Error()
		}
		return "OK", fmt.Sprintf("%s +train (no stores) cells=%d", hop, cells)
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		if hopErr != nil {
			return "GAP", fmt.Sprintf("%s: %v; weights unchanged", hop, hopErr)
		}
		return "GAP", fmt.Sprintf("weights unchanged under %s", mode)
	}
	if hopErr != nil {
		return "GAP", fmt.Sprintf("%s: %v; Δelems=%d max|Δ|=%.6g", hop, hopErr, delta, maxAbs)
	}
	return "OK", fmt.Sprintf("cells=%d Δelems=%d max|Δ|=%.6g", cells, delta, maxAbs)
}
