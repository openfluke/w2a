package gdn

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
	layergdn "github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

var allActKinds = []struct {
	Name string
	Run  func(core.Backend) error
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
	fmt.Printf("\n  GDN activation numeric sweep — %d Tensor[T] kinds × %d backends\n", len(allActKinds), len(backends))
	fmt.Printf("  weights=FormatNone Float32 H=%d T=4  SIMD=%v WebGPU=%v\n\n", matrixCfg().HiddenSize, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-10s %8s  %s\n", "act_T", "backend", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 56))
	var fails []string
	for _, kind := range allActKinds {
		for _, be := range backends {
			status, note := "OK", ""
			if be == core.BackendSIMD && !simd.Enabled() {
				status, note = "GAP", "simd off"
			} else if be == core.BackendWebGPU && !webgpu.Available() {
				status, note = "GAP", "no gpu"
			} else if err := kind.Run(be); err != nil {
				status, note = failOrGap(be), err.Error()
				if status == "FAIL" {
					fails = append(fails, fmt.Sprintf("%s/%s: %s", kind.Name, be, note))
				}
			}
			fmt.Printf("  %-12s %-10s %8s  %s\n", kind.Name, be.String(), status, note)
			rec("acts", kind.Name, "None", be.String(), "-", status, note)
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("activation sweep: %s", strings.Join(fails[:min(8, len(fails))], " | "))
	}
	return nil
}

func smokeAct[T core.Numeric](be core.Backend) error {
	c := matrixCfg()
	l, err := newLayer(c, core.DTypeFloat32, quant.FormatNone, be)
	if err != nil {
		return err
	}
	x := core.NewTensor[T](2, 4, c.HiddenSize)
	for i := range x.Data {
		x.Data[i] = core.FromFloat64[T](float64((i%5)-2) * 0.25)
	}
	pre, post, err := layergdn.Forward(l, x)
	if err != nil {
		return err
	}
	gy := core.NewTensor[T](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = core.FromFloat64[T](1)
	}
	_, _, err = layergdn.Backward(l, gy, x, pre)
	return err
}
