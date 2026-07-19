package dense

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/webgpu"
)

// TimedTrainMatrix — FormatNone × all dtypes × backends on 1×1×1 (single cell).
func TimedTrainMatrix() error {
	return timedTrainFormatNoneGrids([]int{1}, "FormatNone × dtypes × backends (1×1×1)")
}

// TimedTrainQuantMatrix — all quants × backends on 1×1×1.
func TimedTrainQuantMatrix() error {
	return timedTrainQuantGrids([]int{1}, "quants × backends (Float32 pack, 1×1×1)")
}

// TimedTrainGridsFormatNone — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³.
func TimedTrainGridsFormatNone() error {
	return timedTrainFormatNoneGrids([]int{1, 2, 3},
		"FormatNone × ALL dtypes × backends × volumetric 1×1×1 / 2×2×2 / 3×3×3")
}

// TimedTrainGridsQuant — ALL quants × CPU/SIMD/WebGPU × 1³/2³/3³.
func TimedTrainGridsQuant() error {
	return timedTrainQuantGrids([]int{1, 2, 3},
		"ALL quants × backends × volumetric 1×1×1 / 2×2×2 / 3×3×3")
}

func timedTrainFormatNoneGrids(sizes []int, title string) error {
	const (
		dim   = 32
		batch = 2
		lr    = 1e-2
	)
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	total := len(sizes) * len(core.AllDTypes) * len(backends)
	fmt.Printf("\n  Dense TRAIN — %s\n", title)
	fmt.Printf("  cell Dense %d→%d batch=%d  cells=%d  SIMD=%v WebGPU=%v\n\n",
		dim, dim, batch, total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-8s %-12s %-10s %8s %12s %8s  %s\n",
		"grid", "dtype", "backend", "cells", "step_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 84))

	var cpuFails []string
	var okN, gapN int
	for _, n := range sizes {
		warm, iters := trainBudget(n)
		for _, dt := range core.AllDTypes {
			for _, be := range backends {
				ns, status, note := timeTrainCube(n, be, batch, dim, warm, iters, lr, dt, quant.FormatNone)
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
		dim   = 32
		batch = 2
		lr    = 1e-2
	)
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	total := len(sizes) * len(quant.AllFormats) * len(backends)
	fmt.Printf("\n  Dense TRAIN — %s\n", title)
	fmt.Printf("  cell Dense %d→%d batch=%d  cells=%d  SIMD=%v WebGPU=%v\n\n",
		dim, dim, batch, total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-8s %-14s %-10s %8s %12s %8s  %s\n",
		"grid", "format", "backend", "cells", "step_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 86))

	var cpuFails []string
	var okN, gapN int
	for _, n := range sizes {
		warm, iters := trainBudget(n)
		for _, f := range quant.AllFormats {
			for _, be := range backends {
				ns, status, note := timeTrainCube(n, be, batch, dim, warm, iters, lr, core.DTypeFloat32, f)
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
		return 1, 3
	}
}

func timeTrainCube(n int, be core.Backend, batch, dim, warm, iters int, lr float64, dt core.DType, format quant.Format) (ns int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, "GAP", "no gpu"
	}
	g, err := buildDenseCube(n, dim, be, dt, format)
	if err != nil {
		return 0, failOrGap(be), err.Error()
	}
	return benchTrainStep(g, batch, dim, dim, warm, iters, lr, be)
}

func buildDenseCube(n, dim int, be core.Backend, dt core.DType, format quant.Format) (*architecture.Grid, error) {
	g := architecture.NewGrid(n, n, n, 1)
	g.Exec.Backend = be
	base := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		base[i*dim+i] = 1
	}
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				w := append([]float32(nil), base...)
				w[0] += float32(z+y+x) * 0.001
				l, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
				if err != nil {
					return nil, err
				}
				if dt != core.DTypeFloat32 {
					if err := l.Weights.SetDType(dt); err != nil {
						return nil, err
					}
				}
				if format != quant.FormatNone {
					if err := l.Weights.Pack(format); err != nil {
						return nil, err
					}
				}
				if err := dense.Place(g, z, y, x, 0, l); err != nil {
					return nil, err
				}
			}
		}
	}
	return g, nil
}

func benchTrainStep(g *architecture.Grid, batch, in, out, warm, iters int, lr float64, be core.Backend) (ns int64, status, note string) {
	x, y := trainBatch[float32](batch, in, out)
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
		x, y = trainBatch[float32](batch, in, out)
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
	return total.Nanoseconds() / int64(iters), "OK", ""
}

func trainBatch[T core.Numeric](batch, in, out int) (x, y *core.Tensor[T]) {
	x = core.NewTensor[T](batch, in)
	y = core.NewTensor[T](batch, out)
	for b := 0; b < batch; b++ {
		for i := 0; i < in; i++ {
			x.Data[b*in+i] = core.FromFloat64[T](float64((b+i)%5) * 0.1)
		}
		for o := 0; o < out; o++ {
			y.Data[b*out+o] = core.FromFloat64[T](0.5 * float64(b%5) * 0.1)
		}
	}
	return x, y
}

func failOrGap(be core.Backend) string {
	if be == core.BackendCPUTiled {
		return "FAIL"
	}
	return "GAP"
}
