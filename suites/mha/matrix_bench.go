package mha

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

// TimedMatrix — FormatNone × all dtypes × backends, fwd+bwd timings.
func TimedMatrix() error {
	const (
		batch = 2
		seq   = 8
		warm  = 1
		iters = 4
	)
	cfg := defaultCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	fmt.Printf("\n  MHA timed matrix — FormatNone × %d dtypes × %d backends\n", len(core.AllDTypes), len(backends))
	fmt.Printf("  shape batch=%d seq=%d d=%d heads=%d  warm=%d iters=%d  SIMD=%v WebGPU=%v\n\n",
		batch, seq, cfg.DModel, cfg.NumHeads, warm, iters, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", "dtype", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))

	var cpuFails []string
	var okN, gapN int
	for _, dt := range core.AllDTypes {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeCell(dt, quant.FormatNone, be, cfg, batch, seq, warm, iters)
			fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n",
				dt.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
			rec("fwd_bwd", dt.String(), "None", be.String(), "-", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			case "FAIL":
				cpuFails = append(cpuFails, fmt.Sprintf("%s/%s: %s", dt, be, note))
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n",
		okN, gapN, len(cpuFails), len(core.AllDTypes)*len(backends))
	if len(cpuFails) > 0 {
		return fmt.Errorf("timed matrix: %d failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

// TimedQuantMatrix — all quants × backends (Float32 pack).
func TimedQuantMatrix() error {
	const (
		batch = 2
		seq   = 8
		warm  = 1
		iters = 3
	)
	cfg := defaultCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	fmt.Printf("\n  MHA timed quant matrix — %d formats × %d backends (Float32)\n", len(quant.AllFormats), len(backends))
	fmt.Printf("  shape batch=%d seq=%d d=%d  SIMD=%v WebGPU=%v\n\n",
		batch, seq, cfg.DModel, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n", "format", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 80))

	var cpuFails []string
	var okN, gapN int
	for _, f := range quant.AllFormats {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeCell(core.DTypeFloat32, f, be, cfg, batch, seq, warm, iters)
			fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n",
				f.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
			rec("fwd_bwd", "float32", f.String(), be.String(), "-", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			case "FAIL":
				cpuFails = append(cpuFails, fmt.Sprintf("%s/%s: %s", f, be, note))
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n",
		okN, gapN, len(cpuFails), len(quant.AllFormats)*len(backends))
	if len(cpuFails) > 0 {
		return fmt.Errorf("quant matrix: %d failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

func timeCell(dt core.DType, format quant.Format, be core.Backend, cfg mha.Config, batch, seq, warm, iters int) (fwdNs, bwdNs int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, 0, "GAP", "no gpu"
	}
	// Q/K/V weights are [·×DModel]; O is [·×QDim()] — Pack() packs all four,
	// so every projection's column count must be AffinePackable or Pack fails.
	if format == quant.FormatAffinePacked &&
		(!suites.AffinePackable(cfg.QDim(), cfg.DModel) || !suites.AffinePackable(cfg.DModel, cfg.QDim())) {
		return 0, 0, "GAP", suites.AffineSkipNote()
	}
	l, err := newLayer(cfg, dt, format, be)
	if err != nil {
		return 0, 0, failOrGap(be), err.Error()
	}
	x := makeInput(batch, seq, cfg.DModel)
	gOut := core.NewTensor[float32](batch, seq, cfg.DModel)
	for i := range gOut.Data {
		gOut.Data[i] = 1
	}

	var pre *core.Tensor[float32]
	for i := 0; i < warm; i++ {
		var err error
		pre, _, err = mha.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		if _, _, err = mha.Backward(l, gOut, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
	}
	if iters < 1 {
		iters = 1
	}
	var fwdTotal, bwdTotal time.Duration
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		pre, _, err = mha.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		fwdTotal += time.Since(t0)
		t1 := time.Now()
		if _, _, err = mha.Backward(l, gOut, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		bwdTotal += time.Since(t1)
	}
	st, nt := suites.StampBackendNote("mha", be == core.BackendSIMD, be == core.BackendWebGPU, "OK", "")
	return fwdTotal.Nanoseconds() / int64(iters), bwdTotal.Nanoseconds() / int64(iters), st, nt
}

func failOrGap(be core.Backend) string {
	if be == core.BackendCPUTiled {
		return "FAIL"
	}
	return "GAP"
}

func fmtNs(ns int64) string {
	if ns <= 0 {
		return "-"
	}
	if ns < 1e3 {
		return fmt.Sprintf("%dns", ns)
	}
	if ns < 1e6 {
		return fmt.Sprintf("%.1fµs", float64(ns)/1e3)
	}
	if ns < 1e9 {
		return fmt.Sprintf("%.2fms", float64(ns)/1e6)
	}
	return fmt.Sprintf("%.2fs", float64(ns)/1e9)
}
