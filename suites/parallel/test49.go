package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/layers/parallel"
)

// Test49TrainGrids smokes every named TrainMode on 1×1×1 / 2×2×2 / 3×3×3.
// One live cell at the origin (rest disabled) — permutation check, not a full
// volumetric train of n³ copies.
func Test49TrainGrids() error {
	modes := parallel.AllNamedTrainModes()
	if len(modes) < 23 {
		return fmt.Errorf("test49: named modes %d want ≥23", len(modes))
	}
	var fails []string
	fmt.Printf("\n  TEST49 — %d train modes × 1³/2³/3³ origin smoke × Parallel+Bicameral (CPU f32 none)\n", len(modes))
	for _, n := range []int{1, 2, 3} {
		grid := fmt.Sprintf("%dx%dx%d", n, n, n)
		pFail, sFail := 0, 0
		for _, mode := range modes {
			status, note := runCreditParallelCube(n, mode)
			rec("test49_parallel_"+mode.String(), "float32", "none", "cpu_tiled", grid, status, note)
			if status == "FAIL" {
				pFail++
				fails = append(fails, fmt.Sprintf("P%s/%s:%s", grid, mode, note))
				fmt.Printf("  P %-4s %-28s %8s  %s\n", grid, mode.String(), status, note)
			}
			status, note = runCreditStackCube(n, mode)
			rec("test49_stack_"+mode.String(), "float32", "none", "cpu_tiled", grid, status, note)
			if status == "FAIL" {
				sFail++
				fails = append(fails, fmt.Sprintf("S%s/%s:%s", grid, mode, note))
				fmt.Printf("  S %-4s %-28s %8s  %s\n", grid, mode.String(), status, note)
			}
		}
		if pFail == 0 {
			fmt.Printf("  P %-4s %d modes OK\n", grid, len(modes))
		}
		if sFail == 0 {
			fmt.Printf("  S %-4s %d modes OK\n", grid, len(modes))
		}
	}
	if len(fails) > 0 {
		k := min(8, len(fails))
		return fmt.Errorf("test49: %s", strings.Join(fails[:k], "; "))
	}
	return test49PolyKindCubes(modes)
}

// test49PolyKindCubes: every poly layer kind as one cameral stack at the origin
// of 1³/2³/3³ × every named train mode. Hop-chain GAP (embedding tokens, some
// CNN ranks) still trains the placed cell.
func test49PolyKindCubes(modes []parallel.TrainMode) error {
	kinds := polyKinds()
	var fails []string
	var okN, gapN int
	fmt.Printf("\n  TEST49 POLY — %d kinds × %d modes × 1³/2³/3³ (CPU f32 none)\n", len(kinds), len(modes))
	for _, k := range kinds {
		for _, n := range []int{1, 2, 3} {
			g, x, trainable, err := buildPolyCube(k, n)
			grid := fmt.Sprintf("%dx%dx%d", n, n, n)
			if err != nil {
				for _, mode := range modes {
					rec("test49_poly_"+k.name+"_"+mode.String(), "float32", "none", "cpu_tiled", grid, "FAIL", err.Error())
					fmt.Printf("  %-16s %-4s %-28s %8s  %s\n", k.name, grid, mode.String(), "FAIL", err.Error())
					fails = append(fails, fmt.Sprintf("%s/%s/%s:%s", k.name, grid, mode, err))
				}
				continue
			}
			if k.name != "embedding" {
				fillOnes(x)
			}
			y, err := probeStackY(g, x)
			if err != nil {
				for _, mode := range modes {
					rec("test49_poly_"+k.name+"_"+mode.String(), "float32", "none", "cpu_tiled", grid, "FAIL", "probe: "+err.Error())
					fmt.Printf("  %-16s %-4s %-28s %8s  %s\n", k.name, grid, mode.String(), "FAIL", "probe: "+err.Error())
					fails = append(fails, fmt.Sprintf("%s/%s/%s:probe:%s", k.name, grid, mode, err))
				}
				continue
			}
			hopErr := hopGrid(g, x)
			kindFail := 0
			for _, mode := range modes {
				if err := seedGrid(g); err != nil {
					fails = append(fails, fmt.Sprintf("%s/%s/%s:seed:%s", k.name, grid, mode, err))
					kindFail++
					continue
				}
				status, note := trainPlacedCube(g, x, y, mode, 1, trainable, hopErr)
				rec("test49_poly_"+k.name+"_"+mode.String(), "float32", "none", "cpu_tiled", grid, status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				case "FAIL":
					kindFail++
					fails = append(fails, fmt.Sprintf("%s/%s/%s:%s", k.name, grid, mode, note))
					fmt.Printf("  %-16s %-4s %-28s %8s  %s\n", k.name, grid, mode.String(), status, note)
				}
			}
			if kindFail == 0 {
				fmt.Printf("  %-16s %-4s %d modes OK/GAP\n", k.name, grid, len(modes))
			}
		}
	}
	fmt.Printf("  poly summary: %d OK, %d GAP, %d FAIL\n", okN, gapN, len(fails))
	if len(fails) > 0 {
		k := min(8, len(fails))
		return fmt.Errorf("test49 poly: %s", strings.Join(fails[:k], "; "))
	}
	return nil
}
