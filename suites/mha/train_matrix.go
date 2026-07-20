package mha

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/webgpu"
)

// TimedTrainGridsFormatNone — FormatNone × ALL 34 dtypes × backends × 1³/2³/3³.
func TimedTrainGridsFormatNone() error {
	return timedTrainFormatNoneGrids([]int{1, 2, 3},
		"FormatNone × ALL dtypes × backends × volumetric 1×1×1 / 2×2×2 / 3×3×3")
}

// TimedTrainGridsQuant — ALL quants × backends × 1³/2³/3³.
func TimedTrainGridsQuant() error {
	return timedTrainQuantGrids([]int{1, 2, 3},
		"ALL quants × backends × volumetric 1×1×1 / 2×2×2 / 3×3×3")
}

func timedTrainFormatNoneGrids(sizes []int, title string) error {
	const (
		batch = 1
		seq   = 4
		lr    = 1e-2
	)
	cfg := tinyCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	total := len(sizes) * len(core.AllDTypes) * len(backends)

	fmt.Printf("\n  MHA TRAIN — %s\n", title)
	fmt.Printf("  cell MHA d=%d heads=%d seq=%d batch=%d  cells=%d  SIMD=%v WebGPU=%v\n\n",
		cfg.DModel, cfg.NumHeads, seq, batch, total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-8s %-12s %-10s %8s %12s %8s  %s\n",
		"grid", "dtype", "backend", "cells", "step_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 84))

	var cpuFails []string
	var okN, gapN int
	for _, n := range sizes {
		warm, iters := trainBudget(n)
		for _, dt := range core.AllDTypes {
			for _, be := range backends {
				ns, status, note := timeTrainCube(n, be, batch, seq, warm, iters, lr, dt, quant.FormatNone, cfg)
				grid := fmt.Sprintf("%dx%dx%d", n, n, n)
				fmt.Printf("  %-8s %-12s %-10s %8d %12s %8s  %s\n",
					fmt.Sprintf("%d×%d×%d", n, n, n), dt.String(), be.String(), n*n*n, fmtNs(ns), status, note)
				rec("train", dt.String(), "None", be.String(), grid, status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				case "FAIL":
					cpuFails = append(cpuFails, fmt.Sprintf("%dx%dx%d/%s/%s: %s", n, n, n, dt, be, note))
				}
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), total)
	if len(cpuFails) > 0 {
		return fmt.Errorf("train FormatNone grids: %d failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(8, len(cpuFails))], " | "))
	}
	return nil
}

func timedTrainQuantGrids(sizes []int, title string) error {
	const (
		batch = 1
		seq   = 4
		lr    = 1e-2
	)
	cfg := tinyCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	total := len(sizes) * len(quant.AllFormats) * len(backends)

	fmt.Printf("\n  MHA TRAIN — %s\n", title)
	fmt.Printf("  cell MHA d=%d seq=%d batch=%d  cells=%d  SIMD=%v WebGPU=%v\n\n",
		cfg.DModel, seq, batch, total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-8s %-14s %-10s %8s %12s %8s  %s\n",
		"grid", "format", "backend", "cells", "step_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 86))

	var cpuFails []string
	var okN, gapN int
	for _, n := range sizes {
		warm, iters := trainBudget(n)
		for _, f := range quant.AllFormats {
			for _, be := range backends {
				ns, status, note := timeTrainCube(n, be, batch, seq, warm, iters, lr, core.DTypeFloat32, f, cfg)
				grid := fmt.Sprintf("%dx%dx%d", n, n, n)
				fmt.Printf("  %-8s %-14s %-10s %8d %12s %8s  %s\n",
					fmt.Sprintf("%d×%d×%d", n, n, n), f.String(), be.String(), n*n*n, fmtNs(ns), status, note)
				rec("train", "float32", f.String(), be.String(), grid, status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				case "FAIL":
					cpuFails = append(cpuFails, fmt.Sprintf("%dx%dx%d/%s/%s: %s", n, n, n, f, be, note))
				}
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), total)
	if len(cpuFails) > 0 {
		return fmt.Errorf("train quant grids: %d failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(8, len(cpuFails))], " | "))
	}
	return nil
}

func trainBudget(n int) (warm, iters int) {
	switch {
	case n >= 3:
		return 0, 1
	case n == 2:
		return 0, 2
	default:
		return 1, 2
	}
}

func timeTrainCube(n int, be core.Backend, batch, seq, warm, iters int, lr float64, dt core.DType, format quant.Format, cfg mha.Config) (ns int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, "GAP", "no gpu"
	}
	if format == quant.FormatAffinePacked &&
		(!suites.AffinePackable(cfg.QDim(), cfg.DModel) || !suites.AffinePackable(cfg.DModel, cfg.QDim())) {
		return 0, "GAP", suites.AffineSkipNote()
	}
	g, err := buildMHACube(n, be, dt, format, cfg)
	if err != nil {
		return 0, failOrGap(be), err.Error()
	}
	return benchTrainStep(g, batch, seq, cfg.DModel, warm, iters, lr, be)
}

func buildMHACube(n int, be core.Backend, dt core.DType, format quant.Format, cfg mha.Config) (*architecture.Grid, error) {
	g := architecture.NewGrid(n, n, n, 1)
	g.Exec.Backend = be
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				l, err := newLayer(cfg, dt, format, be)
				if err != nil {
					return nil, err
				}
				// tiny perturbation so cells aren't identical
				if w, ok := l.Q.Weights.MasterF32(); ok && len(w) > 0 {
					w[0] += float32(z+y+x) * 0.001
				}
				if err := mha.Place(g, z, y, x, 0, l); err != nil {
					return nil, err
				}
			}
		}
	}
	return g, nil
}

func benchTrainStep(g *architecture.Grid, batch, seq, d, warm, iters int, lr float64, be core.Backend) (ns int64, status, note string) {
	x, y := trainBatch(batch, seq, d)
	for i := 0; i < warm; i++ {
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return 0, failOrGap(be), err.Error()
		}
		if _, err := training.Step(fwd, y, lr); err != nil {
			return 0, failOrGap(be), err.Error()
		}
	}
	if iters < 1 {
		iters = 1
	}
	var total time.Duration
	for i := 0; i < iters; i++ {
		x, y = trainBatch(batch, seq, d)
		t0 := time.Now()
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return 0, failOrGap(be), err.Error()
		}
		if _, err := training.Step(fwd, y, lr); err != nil {
			return 0, failOrGap(be), err.Error()
		}
		total += time.Since(t0)
	}
	st, nt := suites.StampWebGPUNote("mha", be == core.BackendWebGPU, "OK", "")
	return total.Nanoseconds() / int64(iters), st, nt
}

func trainBatch(batch, seq, d int) (x, y *core.Tensor[float32]) {
	x = core.NewTensor[float32](batch, seq, d)
	y = core.NewTensor[float32](batch, seq, d)
	for i := range x.Data {
		x.Data[i] = float32((i%5)-2) * 0.1
		y.Data[i] = float32((i%3)-1) * 0.05
	}
	return x, y
}
