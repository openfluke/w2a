package convt1

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func TimedMatrix() error {
	const batch, warm, iters = 2, 1, 3
	cfg := defaultCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	fmt.Printf("\n  ConvT1 timed matrix — FormatNone × %d dtypes × %d backends\n", len(core.AllDTypes), len(backends))
	fmt.Printf("  shape batch=%d C=%d L=%d→%d  SIMD=%v WebGPU=%v\n\n",
		batch, cfg.InChannels, cfg.SeqLen, cfg.OutLen(), simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", "dtype", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))
	var cpuFails []string
	var okN, gapN int
	for _, dt := range core.AllDTypes {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeCell(dt, quant.FormatNone, be, cfg, batch, warm, iters)
			fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", dt.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
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
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), len(core.AllDTypes)*len(backends))
	if len(cpuFails) > 0 {
		return fmt.Errorf("timed matrix: %d failures: %s", len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

func TimedQuantMatrix() error {
	const batch, warm, iters = 2, 1, 2
	cfg := defaultCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	fmt.Printf("\n  ConvT1 timed quant matrix — %d formats × %d backends (Float32)\n", len(quant.AllFormats), len(backends))
	fmt.Printf("  shape batch=%d C=%d L=%d→%d  SIMD=%v WebGPU=%v\n\n",
		batch, cfg.InChannels, cfg.SeqLen, cfg.OutLen(), simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n", "format", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 80))
	var cpuFails []string
	var okN, gapN int
	for _, f := range quant.AllFormats {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeCell(core.DTypeFloat32, f, be, cfg, batch, warm, iters)
			fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n", f.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
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
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), len(quant.AllFormats)*len(backends))
	if len(cpuFails) > 0 {
		return fmt.Errorf("quant matrix: %d failures: %s", len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

func timeCell(dt core.DType, format quant.Format, be core.Backend, cfg convt1.Config, batch, warm, iters int) (fwdNs, bwdNs int64, status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, 0, "GAP", "no gpu"
	}
	if format == quant.FormatAffinePacked && !affineOK(cfg) {
		return 0, 0, "GAP", suites.AffineSkipNote()
	}
	if !convt1.PermutationOK(dt, format, be) {
		return 0, 0, "GAP", "unsupported permutation"
	}
	l, err := newLayer(cfg, dt, format, be)
	if err != nil {
		return 0, 0, failOrGap(be), err.Error()
	}
	x := makeInput(cfg, batch)
	gOut := core.NewTensor[float32](batch, cfg.Filters, cfg.OutLen())
	for i := range gOut.Data {
		gOut.Data[i] = 1
	}
	var pre *core.Tensor[float32]
	for i := 0; i < warm; i++ {
		pre, _, err = convt1.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		if _, _, err = convt1.Backward(l, gOut, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
	}
	if iters < 1 {
		iters = 1
	}
	var fwdTotal, bwdTotal time.Duration
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		pre, _, err = convt1.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		fwdTotal += time.Since(t0)
		t1 := time.Now()
		if _, _, err = convt1.Backward(l, gOut, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		bwdTotal += time.Since(t1)
	}
	return fwdTotal.Nanoseconds() / int64(iters), bwdTotal.Nanoseconds() / int64(iters), "OK", ""
}
