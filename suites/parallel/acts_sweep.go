package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

var allActKinds = []struct {
	Name string
	Run  func(be core.Backend) error
}{
	{"float32", func(be core.Backend) error { return smokeAct[float32](be) }},
	{"float64", func(be core.Backend) error { return smokeAct[float64](be) }},
	{"int", func(be core.Backend) error { return smokeAct[int](be) }},
	{"int8", func(be core.Backend) error { return smokeAct[int8](be) }},
	{"int16", func(be core.Backend) error { return smokeAct[int16](be) }},
	{"int32", func(be core.Backend) error { return smokeAct[int32](be) }},
	{"int64", func(be core.Backend) error { return smokeAct[int64](be) }},
	{"uint", func(be core.Backend) error { return smokeAct[uint](be) }},
	{"uint8", func(be core.Backend) error { return smokeAct[uint8](be) }},
	{"uint16", func(be core.Backend) error { return smokeAct[uint16](be) }},
	{"uint32", func(be core.Backend) error { return smokeAct[uint32](be) }},
	{"uint64", func(be core.Backend) error { return smokeAct[uint64](be) }},
	{"uintptr", func(be core.Backend) error { return smokeAct[uintptr](be) }},
	{"complex64", func(be core.Backend) error { return smokeAct[complex64](be) }},
	{"complex128", func(be core.Backend) error { return smokeAct[complex128](be) }},
}

func ActNumericSweep() error {
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	total := len(allActKinds) * len(backends)
	fmt.Printf("\n  Parallel activation numeric sweep — %d Tensor[T] kinds × %d backends\n", len(allActKinds), len(backends))
	fmt.Printf("  weights=FormatNone Float32  Dim=%d  SIMD=%v WebGPU=%v\n\n",
		tinyCfg().Dim, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %8s  %s\n", "act_T", "backend", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 56))
	var cpuFails []string
	var okN, gapN int
	for _, ak := range allActKinds {
		for _, be := range backends {
			status, note := "OK", ""
			if be == core.BackendSIMD && !simd.Enabled() {
				status, note = "GAP", "simd off"
			} else if be == core.BackendWebGPU && !webgpu.Available() {
				status, note = "GAP", "no gpu"
			} else if err := ak.Run(be); err != nil {
				status, note = failOrGap(be), err.Error()
			}
			fmt.Printf("  %-12s %-10s %8s  %s\n", ak.Name, be.String(), status, note)
			rec("acts", ak.Name, "None", be.String(), "-", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			case "FAIL":
				cpuFails = append(cpuFails, fmt.Sprintf("%s/%s: %s", ak.Name, be, note))
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), total)
	if len(cpuFails) > 0 {
		return fmt.Errorf("act numeric sweep: %d failures: %s", len(cpuFails), strings.Join(cpuFails[:min(8, len(cpuFails))], " | "))
	}
	return nil
}

func smokeAct[T core.Numeric](be core.Backend) error {
	cfg := tinyCfg()
	packed := make([]T, cfg.Branches*cfg.OutFeat*cfg.Dim)
	for i := range packed {
		packed[i] = core.FromFloat64[T](float64((i%5)-2) * 0.02)
	}
	l, err := parallel.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, packed, nil)
	if err != nil {
		return err
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	x := core.NewTensor[T](2, cfg.Dim)
	for i := range x.Data {
		x.Data[i] = core.FromFloat64[T](float64((i%5)-2) * 0.25)
	}
	pre, post, err := parallel.Forward(l, x)
	if err != nil {
		return fmt.Errorf("fwd: %w", err)
	}
	g := core.NewTensor[T](post.Shape...)
	for i := range g.Data {
		g.Data[i] = core.FromFloat64[T](1)
	}
	_, _, err = parallel.Backward(l, g, x, pre)
	if err != nil {
		return fmt.Errorf("bwd: %w", err)
	}
	return nil
}
