package gdn

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/welvet/core"
	layergdn "github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func TimedMatrix() error {
	const batch, warm, iters = 2, 1, 3
	c := matrixCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	fmt.Printf("\n  GDN timed matrix — FormatNone × %d dtypes × %d backends\n", len(core.AllDTypes), len(backends))
	fmt.Printf("  shape batch=%d H=%d T=4 heads=2 dims=8  SIMD=%v WebGPU=%v\n\n", batch, c.HiddenSize, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", "dtype", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))
	return timedCells(core.AllDTypes, []quant.Format{quant.FormatNone}, backends, c, batch, warm, iters)
}

func TimedQuantMatrix() error {
	const batch, warm, iters = 2, 1, 2
	c := matrixCfg()
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	formats := []quant.Format{quant.FormatNone, quant.FormatBinaryPacked}
	fmt.Printf("\n  GDN timed quant matrix — None/BinaryPacked × %d backends\n", len(backends))
	fmt.Printf("  shape batch=%d H=%d T=4 heads=2 dims=8  SIMD=%v WebGPU=%v\n\n", batch, c.HiddenSize, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", "format", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))
	return timedCells([]core.DType{core.DTypeFloat32}, formats, backends, c, batch, warm, iters)
}

func timedCells(dtypes []core.DType, formats []quant.Format, backends []core.Backend, c layergdn.Config, batch, warm, iters int) error {
	var fails []string
	var okN, gapN int
	for _, dt := range dtypes {
		for _, format := range formats {
			for _, be := range backends {
				fwdNs, bwdNs, status, note := timeCell(dt, format, be, c, batch, warm, iters)
				label := dt.String()
				if len(formats) > 1 {
					label = format.String()
				}
				fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", label, be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
				rec("fwd_bwd", dt.String(), format.String(), be.String(), "-", status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				case "FAIL":
					fails = append(fails, fmt.Sprintf("%s/%s/%s: %s", dt, format, be, note))
				}
			}
		}
	}
	total := len(dtypes) * len(formats) * len(backends)
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(fails), total)
	if len(fails) > 0 {
		return fmt.Errorf("timed matrix: %d failures: %s", len(fails), strings.Join(fails[:min(6, len(fails))], " | "))
	}
	return nil
}

func timeCell(dt core.DType, format quant.Format, be core.Backend, c layergdn.Config, batch, warm, iters int) (fwdNs, bwdNs int64, status, note string) {
	if !layergdn.PermutationOK(dt, format, be) {
		return 0, 0, "GAP", "unsupported GDN permutation"
	}
	if be == core.BackendSIMD && !simd.Enabled() {
		return 0, 0, "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return 0, 0, "GAP", "no gpu"
	}
	l, err := newLayer(c, dt, format, be)
	if err != nil {
		return 0, 0, failOrGap(be), err.Error()
	}
	x := makeInput(c, batch)
	gy := core.NewTensor[float32](batch, 4, c.HiddenSize)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	for i := 0; i < warm; i++ {
		pre, _, err := layergdn.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		if _, _, err := layergdn.Backward(l, gy, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
	}
	var fwdTotal, bwdTotal time.Duration
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		pre, _, err := layergdn.Forward(l, x)
		if err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		fwdTotal += time.Since(t0)
		t1 := time.Now()
		if _, _, err := layergdn.Backward(l, gy, x, pre); err != nil {
			return 0, 0, failOrGap(be), err.Error()
		}
		bwdTotal += time.Since(t1)
	}
	return fwdTotal.Nanoseconds() / int64(iters), bwdTotal.Nanoseconds() / int64(iters), "OK", ""
}

func fmtNs(ns int64) string {
	switch {
	case ns <= 0:
		return "-"
	case ns < 1e3:
		return fmt.Sprintf("%dns", ns)
	case ns < 1e6:
		return fmt.Sprintf("%.1fµs", float64(ns)/1e3)
	case ns < 1e9:
		return fmt.Sprintf("%.2fms", float64(ns)/1e6)
	default:
		return fmt.Sprintf("%.2fs", float64(ns)/1e9)
	}
}
