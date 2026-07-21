package seven

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

const sevenLayersPerCell = 7

func buildSevenDenseCube(n, dim int, multi bool) (*architecture.Grid, error) {
	g := architecture.NewGrid(n, n, n, sevenLayersPerCell)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = multi
	g.Exec.TileSize = 32
	base := makeDenseInit(dim, dim)
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				for l := 0; l < sevenLayersPerCell; l++ {
					w := append([]float32(nil), base...)
					w[0] += float32(z+y+x+l) * 0.001
					layer, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
					if err != nil {
						return nil, err
					}
					if err := dense.Place(g, z, y, x, l, layer); err != nil {
						return nil, err
					}
				}
			}
		}
	}
	return g, nil
}

func volumetricSevenDense() error {
	const dim = 64
	x, y := makeDenseXY(2, dim, dim)
	for _, n := range []int{1, 2, 3} {
		gSC, err := buildSevenDenseCube(n, dim, false)
		if err != nil {
			return fmt.Errorf("%d³ SC build: %w", n, err)
		}
		gMC, err := buildSevenDenseCube(n, dim, true)
		if err != nil {
			return fmt.Errorf("%d³ MC build: %w", n, err)
		}
		fwdSC, err := forward.Forward(gSC, x)
		if err != nil {
			return fmt.Errorf("%d³ SC fwd: %w", n, err)
		}
		fwdMC, err := forward.Forward(gMC, x)
		if err != nil {
			return fmt.Errorf("%d³ MC fwd: %w", n, err)
		}
		dP := suites.MaxAbsDiff(fwdSC.Output.Data, fwdMC.Output.Data)
		if err := suites.RequireDet(fmt.Sprintf("%d³ fwd SC↔MC", n), dP, suites.DetTolFwd); err != nil {
			return err
		}
		// Short train only on 1³ — larger cubes (56–189 Dense ops) blow up at default lr.
		if n == 1 {
			first, last, err := trainEpochs(gSC, x, y, 3, 1e-3)
			if err != nil {
				return fmt.Errorf("%d³ train: %w", n, err)
			}
			fmt.Printf("(%d³ Δ=%.3g loss %.4g→%.4g) ", n, dP, first, last)
			continue
		}
		fmt.Printf("(%d³ Δ=%.3g sc↔mc) ", n, dP)
	}
	return nil
}
