package dense

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

// TimedMatrix runs FormatNone × all dtypes × {CPU,SIMD,WebGPU} with fwd/bwd timings.
// Prints a full performance table. Cells that hard-error are reported as GAP (not FAIL)
// unless CPU FormatNone fails (that must work for every dtype).
func TimedMatrix() error {
	const (
		in     = 256
		out    = 128
		batch  = 8
		warm   = 2
		iters  = 12
	)
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	fmt.Printf("\n  Dense timed matrix — FormatNone × %d dtypes × %d backends\n", len(core.AllDTypes), len(backends))
	gpuNote := "no"
	if webgpu.Available() {
		gpuNote = webgpu.AdapterName()
		if gpuNote == "" {
			gpuNote = "yes"
		}
	}
	fmt.Printf("  shape batch=%d in=%d out=%d  warm=%d iters=%d  SIMD=%v WebGPU=%s\n\n",
		batch, in, out, warm, iters, simd.Enabled(), gpuNote)
	fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n", "dtype", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))

	var cpuFails []string
	var okN, gapN int

	for _, dt := range core.AllDTypes {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeCell(dt, be, batch, in, out, warm, iters)
			fmt.Printf("  %-12s %-10s %10s %10s %8s  %s\n",
				dt.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
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
		return fmt.Errorf("timed matrix: %d CPU/required failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

func timeCell(dt core.DType, be core.Backend, batch, in, out, warm, iters int) (fwdNs, bwdNs int64, status, note string) {
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
	if err != nil {
		return 0, 0, "FAIL", err.Error()
	}
	if dt != core.DTypeFloat32 {
		if err := l.Weights.SetDType(dt); err != nil {
			if be == core.BackendCPUTiled {
				return 0, 0, "FAIL", "SetDType: " + err.Error()
			}
			return 0, 0, "GAP", "SetDType: " + err.Error()
		}
		l.Core.DType = dt
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false

	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.25
	}
	gOut := core.NewTensor[float32](batch, out)
	for i := range gOut.Data {
		gOut.Data[i] = 1
	}

	// Warmup + forward timing
	var pre *core.Tensor[float32]
	for i := 0; i < warm; i++ {
		pre, _, err = dense.Forward(l, x)
		if err != nil {
			break
		}
	}
	if err != nil {
		if be == core.BackendCPUTiled {
			return 0, 0, "FAIL", "fwd: " + err.Error()
		}
		return 0, 0, "GAP", trimErr(err)
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		pre, _, err = dense.Forward(l, x)
		if err != nil {
			break
		}
	}
	fwdNs = time.Since(t0).Nanoseconds() / int64(iters)
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, 0, "FAIL", "fwd: " + err.Error()
		}
		return fwdNs, 0, "GAP", trimErr(err)
	}

	for i := 0; i < warm; i++ {
		_, _, err = dense.Backward(l, gOut, x, pre)
		if err != nil {
			break
		}
	}
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, 0, "FAIL", "bwd: " + err.Error()
		}
		return fwdNs, 0, "GAP", "bwd: " + trimErr(err)
	}

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		_, _, err = dense.Backward(l, gOut, x, pre)
		if err != nil {
			break
		}
	}
	bwdNs = time.Since(t1).Nanoseconds() / int64(iters)
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, bwdNs, "FAIL", "bwd: " + err.Error()
		}
		return fwdNs, bwdNs, "GAP", "bwd: " + trimErr(err)
	}
	return fwdNs, bwdNs, "OK", ""
}

// TimedQuantMatrix runs all quant.Formats × {CPU,SIMD,WebGPU} at Float32 with timings.
func TimedQuantMatrix() error {
	const (
		in    = 256
		out   = 128
		batch = 4
		warm  = 2
		iters = 8
	)
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}

	fmt.Printf("\n  Dense timed quant matrix — %d formats × %d backends (Float32)\n", len(quant.AllFormats), len(backends))
	gpuNote := "no"
	if webgpu.Available() {
		gpuNote = webgpu.AdapterName()
		if gpuNote == "" {
			gpuNote = "yes"
		}
	}
	fmt.Printf("  shape batch=%d in=%d out=%d  warm=%d iters=%d  SIMD=%v WebGPU=%s\n\n",
		batch, in, out, warm, iters, simd.Enabled(), gpuNote)
	fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n", "format", "backend", "fwd_ns/op", "bwd_ns/op", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 80))

	var cpuFails []string
	var okN, gapN int

	for _, f := range quant.AllFormats {
		for _, be := range backends {
			fwdNs, bwdNs, status, note := timeQuantCell(f, be, batch, in, out, warm, iters)
			fmt.Printf("  %-14s %-10s %10s %10s %8s  %s\n",
				f.String(), be.String(), fmtNs(fwdNs), fmtNs(bwdNs), status, note)
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
		return fmt.Errorf("timed quant matrix: %d CPU failures: %s",
			len(cpuFails), strings.Join(cpuFails[:min(6, len(cpuFails))], " | "))
	}
	return nil
}

func timeQuantCell(f quant.Format, be core.Backend, batch, in, out, warm, iters int) (fwdNs, bwdNs int64, status, note string) {
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, f, init)
	if err != nil {
		if be == core.BackendCPUTiled {
			return 0, 0, "FAIL", err.Error()
		}
		return 0, 0, "GAP", trimErr(err)
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false

	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5) * 0.25
	}
	gOut := core.NewTensor[float32](batch, out)
	for i := range gOut.Data {
		gOut.Data[i] = 1
	}

	var pre *core.Tensor[float32]
	for i := 0; i < warm; i++ {
		pre, _, err = dense.Forward(l, x)
		if err != nil {
			break
		}
	}
	if err != nil {
		if be == core.BackendCPUTiled {
			return 0, 0, "FAIL", "fwd: " + err.Error()
		}
		return 0, 0, "GAP", trimErr(err)
	}

	t0 := time.Now()
	for i := 0; i < iters; i++ {
		pre, _, err = dense.Forward(l, x)
		if err != nil {
			break
		}
	}
	fwdNs = time.Since(t0).Nanoseconds() / int64(iters)
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, 0, "FAIL", "fwd: " + err.Error()
		}
		return fwdNs, 0, "GAP", trimErr(err)
	}

	for i := 0; i < warm; i++ {
		_, _, err = dense.Backward(l, gOut, x, pre)
		if err != nil {
			break
		}
	}
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, 0, "FAIL", "bwd: " + err.Error()
		}
		return fwdNs, 0, "GAP", "bwd: " + trimErr(err)
	}

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		_, _, err = dense.Backward(l, gOut, x, pre)
		if err != nil {
			break
		}
	}
	bwdNs = time.Since(t1).Nanoseconds() / int64(iters)
	if err != nil {
		if be == core.BackendCPUTiled {
			return fwdNs, bwdNs, "FAIL", "bwd: " + err.Error()
		}
		return fwdNs, bwdNs, "GAP", "bwd: " + trimErr(err)
	}
	note = ""
	if f == quant.FormatQ4_0 && be == core.BackendSIMD {
		note = "fused DotQ4_0 when aligned"
  } else if f != quant.FormatNone && be != core.BackendCPUTiled {
		note = "quant→host wire (ggml f32 unpack→MAC)"
	} else if f == quant.FormatNone && be == core.BackendSIMD {
		note = "SelectWire F32/F64/I8"
	}
	return fwdNs, bwdNs, "OK", note
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

func trimErr(err error) string {
	s := err.Error()
	if len(s) > 48 {
		return s[:48] + "…"
	}
	return s
}
