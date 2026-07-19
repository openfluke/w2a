package tween

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/training"
)

const (
	timedDim   = 256 // large enough for DotTile to beat scalar CPU
	timedWarm  = 2
	timedIters = 8
	timedLR    = 0.01
)

// TimedDenseFormatNone — Dense StepTween CPU vs SIMD across all 34 dtypes.
func TimedDenseFormatNone() error {
	return timedDenseCompare(
		"FormatNone × all 34 dtypes (Dense %d→%d)",
		func(yield func(dt core.DType, format quant.Format, label string)) {
			for _, dt := range core.AllDTypes {
				yield(dt, quant.FormatNone, dt.String())
			}
		},
	)
}

// TimedDenseQuants — Dense StepTween CPU vs SIMD across all quants (Float32).
func TimedDenseQuants() error {
	return timedDenseCompare(
		"all quants × Float32 (Dense %d→%d)",
		func(yield func(dt core.DType, format quant.Format, label string)) {
			for _, f := range quant.AllFormats {
				yield(core.DTypeFloat32, f, f.String())
			}
		},
	)
}

// TimedLayersCPUVsSIMD — every layer FormatNone Float32, CPU vs SIMD side-by-side.
func TimedLayersCPUVsSIMD() error {
	kinds := polyops.AllKinds()
	fmt.Printf("\n  Tween TIMED layers — FormatNone Float32 × %d layers × CPU vs SIMD\n", len(kinds))
	fmt.Printf("  warm=%d iters=%d  SIMD=%v\n\n", timedWarm, timedIters, simd.Enabled())
	fmt.Printf("  %-12s %12s %12s %10s %8s  %s\n",
		"layer", "cpu_ns/op", "simd_ns/op", "speedup", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))

	var fails []string
	var okN, gapN int
	for _, k := range kinds {
		cpuNs, cpuSt, cpuNote := timeKindStep(k, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
		simdNs, simdSt, simdNote := timeKindStep(k, core.DTypeFloat32, quant.FormatNone, core.BackendSIMD)
		status, note, speed := pairStatus(cpuNs, cpuSt, cpuNote, simdNs, simdSt, simdNote)
		fmt.Printf("  %-12s %12s %12s %10s %8s  %s\n",
			k.Name, fmtNs(cpuNs), fmtNs(simdNs), speed, status, note)
		recTimed(k.Name, "f32", "none", "cpu", cpuSt, cpuNote)
		recTimed(k.Name, "f32", "none", "simd", simdSt, simdNote)
		switch status {
		case "OK":
			okN++
		case "GAP":
			gapN++
		case "FAIL":
			fails = append(fails, fmt.Sprintf("%s: %s", k.Name, note))
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d layers)\n", okN, gapN, len(fails), len(kinds))
	if len(fails) > 0 {
		return fmt.Errorf("timed layers: %d failures: %s", len(fails), strings.Join(fails[:min(6, len(fails))], " | "))
	}
	return nil
}

func timedDenseCompare(titleFmt string, each func(func(dt core.DType, format quant.Format, label string))) error {
	fmt.Printf("\n  Tween TIMED Dense CPU vs SIMD — "+titleFmt+"\n", timedDim, timedDim)
	fmt.Printf("  warm=%d iters=%d  SIMD=%v\n\n", timedWarm, timedIters, simd.Enabled())
	fmt.Printf("  %-14s %12s %12s %10s %8s  %s\n",
		"cell", "cpu_ns/op", "simd_ns/op", "speedup", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 80))

	var fails []string
	var okN, gapN, total int
	each(func(dt core.DType, format quant.Format, label string) {
		total++
		cpuNs, cpuSt, cpuNote := timeDenseStep(dt, format, core.BackendCPUTiled)
		simdNs, simdSt, simdNote := timeDenseStep(dt, format, core.BackendSIMD)
		status, note, speed := pairStatus(cpuNs, cpuSt, cpuNote, simdNs, simdSt, simdNote)
		fmt.Printf("  %-14s %12s %12s %10s %8s  %s\n",
			label, fmtNs(cpuNs), fmtNs(simdNs), speed, status, note)
		recTimed("dense", dt.String(), format.String(), "cpu", cpuSt, cpuNote)
		recTimed("dense", dt.String(), format.String(), "simd", simdSt, simdNote)
		switch status {
		case "OK":
			okN++
		case "GAP":
			gapN++
		case "FAIL":
			fails = append(fails, fmt.Sprintf("%s: %s", label, note))
		}
	})
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(fails), total)
	if len(fails) > 0 {
		return fmt.Errorf("timed dense: %d failures: %s", len(fails), strings.Join(fails[:min(6, len(fails))], " | "))
	}
	return nil
}

func pairStatus(cpuNs int64, cpuSt, cpuNote string, simdNs int64, simdSt, simdNote string) (status, note, speed string) {
	speed = "-"
	if cpuSt == "OK" && simdSt == "OK" && cpuNs > 0 && simdNs > 0 {
		speed = fmt.Sprintf("%.2fx", float64(cpuNs)/float64(simdNs))
		return "OK", "", speed
	}
	if cpuSt == "FAIL" || simdSt == "FAIL" {
		note = cpuNote
		if cpuSt != "FAIL" {
			note = "simd: " + simdNote
		} else if cpuNote != "" {
			note = "cpu: " + cpuNote
		}
		return "FAIL", note, speed
	}
	// GAP on construct/pack (e.g. AffinePacked) or simd off
	if cpuSt == "GAP" && simdSt == "GAP" {
		note = cpuNote
		if note == "" {
			note = simdNote
		}
		return "GAP", note, speed
	}
	if cpuSt == "OK" && simdSt == "GAP" {
		return "OK", "simd: " + simdNote, speed
	}
	if cpuSt == "GAP" && simdSt == "OK" {
		return "OK", "cpu: " + cpuNote, speed
	}
	return "GAP", cpuNote, speed
}

func timeDenseStep(dt core.DType, format quant.Format, be core.Backend) (ns int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, "GAP", "simd off"
	}
	g, err := buildTimedDense(dt, format, be)
	if err != nil {
		return 0, "GAP", trim(err.Error())
	}
	x := core.NewTensor[float32](1, timedDim)
	target := core.NewTensor[float32](1, timedDim)
	for i := 0; i < timedDim; i++ {
		x.Data[i] = float32((i%7)+1) * 0.05
		target.Data[i] = float32((i%7)+1) * 0.06
	}
	return benchStepTween(g, x, target, be)
}

func buildTimedDense(dt core.DType, format quant.Format, be core.Backend) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	g.Exec.Backend = be
	w := make([]float32, timedDim*timedDim)
	for i := 0; i < timedDim; i++ {
		w[i*timedDim+i] = 1
	}
	l, err := dense.NewConfigured(timedDim, timedDim, core.ActivationLinear, dt, format, w)
	if err != nil {
		return nil, err
	}
	if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
		return nil, err
	}
	return g, nil
}

func timeKindStep(k polyops.Kind, dt core.DType, format quant.Format, be core.Backend) (ns int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, "GAP", "simd off"
	}
	g, err := polyops.MakeBackend(k, dt, format, be)
	if err != nil {
		if be == core.BackendCPUTiled {
			return 0, "FAIL", trim(err.Error())
		}
		return 0, "GAP", trim(err.Error())
	}
	x, target := polyops.MakeIO(k, 1.15)
	return benchStepTween(g, x, target, be)
}

func benchStepTween(g *architecture.Grid, x, target *core.Tensor[float32], be core.Backend) (ns int64, status, note string) {
	for i := 0; i < timedWarm; i++ {
		if _, _, err := training.StepTween(g, x, target, timedLR); err != nil {
			if be == core.BackendCPUTiled {
				return 0, "FAIL", trim(err.Error())
			}
			return 0, "GAP", trim(err.Error())
		}
	}
	var total time.Duration
	for i := 0; i < timedIters; i++ {
		t0 := time.Now()
		if _, _, err := training.StepTween(g, x, target, timedLR); err != nil {
			if be == core.BackendCPUTiled {
				return 0, "FAIL", trim(err.Error())
			}
			return 0, "GAP", trim(err.Error())
		}
		total += time.Since(t0)
	}
	return total.Nanoseconds() / int64(timedIters), "OK", ""
}

func recTimed(op, dt, format, backend, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "tween", Op: op, DType: dt, Format: format, Backend: backend,
		Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func fmtNs(ns int64) string {
	if ns <= 0 {
		return "-"
	}
	if ns < 1_000 {
		return fmt.Sprintf("%dns", ns)
	}
	if ns < 1_000_000 {
		return fmt.Sprintf("%.1fµs", float64(ns)/1e3)
	}
	return fmt.Sprintf("%.2fms", float64(ns)/1e6)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
