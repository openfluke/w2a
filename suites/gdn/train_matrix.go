package gdn

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	layergdn "github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func TimedTrainGridsFormatNone() error {
	return timedTrain([]core.DType{}, []quant.Format{quant.FormatNone}, "FormatNone × dtypes × backends × 1³/2³/3³")
}

func TimedTrainGridsQuant() error {
	return timedTrain([]core.DType{core.DTypeFloat32}, []quant.Format{quant.FormatNone, quant.FormatBinaryPacked}, "None/BinaryPacked × backends × 1³/2³/3³")
}

func timedTrain(dtypes []core.DType, formats []quant.Format, title string) error {
	if len(dtypes) == 0 {
		dtypes = core.AllDTypes
	}
	const batch, lr = 2, 1e-2
	c := matrixCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	total := 3 * len(dtypes) * len(formats) * len(backends)
	fmt.Printf("\n  GDN TRAIN — %s\n", title)
	fmt.Printf("  cell GDN H=%d T=4 batch=%d cells=%d SIMD=%v WebGPU=%v\n\n", c.HiddenSize, batch, total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-8s %-12s %-12s %-10s %12s %8s  %s\n", "grid", "dtype", "format", "backend", "step_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 96))
	var fails []string
	var okN, gapN int
	for _, n := range []int{1, 2, 3} {
		warm, iters := trainBudget(n)
		for _, dt := range dtypes {
			for _, format := range formats {
				for _, be := range backends {
					ns, status, note := timeTrainCube(n, be, batch, warm, iters, lr, dt, format, c)
					grid := fmt.Sprintf("%d×%d×%d", n, n, n)
					fmt.Printf("  %-8s %-12s %-12s %-10s %12s %8s  %s\n", grid, dt, format, be, fmtNs(ns), status, note)
					rec("train", dt.String(), format.String(), be.String(), fmt.Sprintf("%dx%dx%d", n, n, n), status, note)
					switch status {
					case "OK":
						okN++
					case "GAP":
						gapN++
					case "FAIL":
						fails = append(fails, fmt.Sprintf("%s/%s/%s: %s", grid, dt, be, note))
					}
				}
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(fails), total)
	if len(fails) > 0 {
		return fmt.Errorf("train matrix: %s", strings.Join(fails[:min(8, len(fails))], " | "))
	}
	return nil
}

func trainBudget(n int) (warm, iters int) {
	if n >= 3 {
		return 0, 1
	}
	if n == 2 {
		return 0, 2
	}
	return 1, 2
}

func timeTrainCube(n int, be core.Backend, batch, warm, iters int, lr float64, dt core.DType, format quant.Format, c layergdn.Config) (int64, string, string) {
	if !layergdn.PermutationOK(dt, format, be) {
		return 0, "GAP", "unsupported GDN permutation"
	}
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, "GAP", "no gpu"
	}
	g, err := buildGDNCube(n, be, dt, format, c)
	if err != nil {
		return 0, failOrGap(be), err.Error()
	}
	x, target := trainBatch(batch, c)
	for i := 0; i < warm; i++ {
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return 0, failOrGap(be), err.Error()
		}
		if _, err := training.Step(fwd, target, lr); err != nil {
			return 0, failOrGap(be), err.Error()
		}
	}
	var total time.Duration
	for i := 0; i < iters; i++ {
		x, target = trainBatch(batch, c)
		t0 := time.Now()
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return 0, failOrGap(be), err.Error()
		}
		if _, err := training.Step(fwd, target, lr); err != nil {
			return 0, failOrGap(be), err.Error()
		}
		total += time.Since(t0)
	}
	return total.Nanoseconds() / int64(iters), "OK", ""
}

func buildGDNCube(n int, be core.Backend, dt core.DType, format quant.Format, c layergdn.Config) (*architecture.Grid, error) {
	g := architecture.NewGrid(n, n, n, 1)
	g.Exec.Backend = be
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				l, err := newLayer(c, dt, format, be)
				if err != nil {
					return nil, err
				}
				if err := layergdn.Place(g, z, y, x, 0, l); err != nil {
					return nil, err
				}
			}
		}
	}
	return g, nil
}

func trainBatch(batch int, c layergdn.Config) (x, target *core.Tensor[float32]) {
	x = core.NewTensor[float32](batch, 4, c.HiddenSize)
	target = core.NewTensor[float32](batch, 4, c.HiddenSize)
	for i := range x.Data {
		x.Data[i] = float32((i%5)-2) * 0.1
		target.Data[i] = float32((i%3)-1) * 0.05
	}
	return x, target
}
