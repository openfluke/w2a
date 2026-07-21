package layernorm

import (
	"fmt"
	"math"
	"runtime"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func lnSimdScaleParity() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	cfg := tinyCfg()
	x := makeInput(2, cfg.Dim)
	run := func(be core.Backend) (*core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, err
		}
		_, post, err := layernorm.Forward(l, x)
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
		return fmt.Errorf("LN SIMD scale maxΔ=%g", max)
	}
	fmt.Printf("(LayerNormScaleF32 Δ=%.3g) ", max)
	return nil
}

func layerNormWebGPUBwdParity() error {
	if !webgpu.Available() {
		fmt.Printf("(no GPU — skip) ")
		return nil
	}
	cfg := tinyCfg()
	x := makeInput(2, cfg.Dim)
	run := func(be core.Backend) (*core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, nil, err
		}
		pre, post, err := layernorm.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		gIn, gW, err := layernorm.Backward(l, g, x, pre)
		return gIn, gW, err
	}
	dxCPU, dwCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	dxGPU, dwGPU, err := run(core.BackendWebGPU)
	if err != nil {
		return err
	}
	maxX, maxW := 0.0, 0.0
	for i := range dxCPU.Data {
		e := math.Abs(float64(dxCPU.Data[i] - dxGPU.Data[i]))
		if e > maxX {
			maxX = e
		}
	}
	for i := range dwCPU.Data {
		e := math.Abs(float64(dwCPU.Data[i] - dwGPU.Data[i]))
		if e > maxW {
			maxW = e
		}
	}
	const tol = 2e-2
	if maxX > tol || maxW > tol {
		return fmt.Errorf("LN WebGPU bwd dxΔ=%g dWΔ=%g", maxX, maxW)
	}
	fmt.Printf("(LN bwd GPU dxΔ=%.3g dWΔ=%.3g) ", maxX, maxW)
	return nil
}
