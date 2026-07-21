package softmax

import (
	"fmt"
	"math"
	"runtime"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func exoticWebGPUSmoke() error {
	if !webgpu.Available() {
		fmt.Printf("(no GPU — skip) ")
		return nil
	}
	kinds := []struct {
		k     softmax.Kind
		extra func(*softmax.Config)
	}{
		{softmax.KindGumbel, nil},
		{softmax.KindMasked, func(c *softmax.Config) {
			c.Mask = []bool{true, true, false, true, true, true, true, true}
		}},
		{softmax.KindSparse, nil},
		{softmax.KindEntmax, func(c *softmax.Config) { c.EntmaxAlpha = 1.5 }},
	}
	for _, item := range kinds {
		cfg := tinyCfg()
		cfg.Kind = item.k
		if item.extra != nil {
			item.extra(&cfg)
		}
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendWebGPU)
		if err != nil {
			return fmt.Errorf("%s new: %w", item.k, err)
		}
		x := makeInput(cfg, 1)
		pre, post, err := softmax.Forward(l, x)
		if err != nil {
			return fmt.Errorf("%s fwd: %w", item.k, err)
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		if _, _, err := softmax.Backward(l, g, x, pre); err != nil {
			return fmt.Errorf("%s bwd: %w", item.k, err)
		}
	}
	fmt.Printf("(Gumbel/Masked/Sparse/Entmax WebGPU) ")
	return nil
}

func simdSoftmaxParity() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	cfg := tinyCfg()
	cfg.Kind = softmax.KindStandard
	x := makeInput(cfg, 2)
	run := func(be core.Backend) (*core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, err
		}
		_, post, err := softmax.Forward(l, x)
		return post, err
	}
	pCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	pSIMD, err := run(core.BackendSIMD)
	if err != nil {
		return err
	}
	var max float64
	for i := range pCPU.Data {
		e := math.Abs(float64(pCPU.Data[i] - pSIMD.Data[i]))
		if e > max {
			max = e
		}
	}
	if max > 1e-4 {
		return fmt.Errorf("CPU↔SIMD Softmax maxΔ=%g", max)
	}
	row := x.Data[:cfg.Dim]
	y := make([]float32, cfg.Dim)
	simd.SoftmaxF32(row, y, cfg.Dim, 1)
	var sum float64
	for _, v := range y {
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-4 {
		return fmt.Errorf("SoftmaxF32 row sum=%g", sum)
	}
	fmt.Printf("(SoftmaxF32 parity Δ=%.3g) ", max)
	return nil
}
