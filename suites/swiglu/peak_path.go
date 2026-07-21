package swiglu

import (
	"fmt"
	"math"
	"runtime"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

func siluMulSIMDParity() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	return cpuVsBackendParity(core.BackendSIMD, 5e-3, "SIMD SiluMul")
}

func swigluFuseBwdWebGPUParity() error {
	if !webgpu.Available() {
		fmt.Printf("(no GPU — skip) ")
		return nil
	}
	return cpuVsBackendParity(core.BackendWebGPU, 2e-2, "WebGPU fuse bwd")
}

func cpuVsBackendParity(be core.Backend, tol float64, label string) error {
	cfg := tinyCfg()
	x := makeInput(2, cfg.InputDim)
	run := func(b core.Backend) (*core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, b)
		if err != nil {
			return nil, nil, err
		}
		pre, post, err := swiglu.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dW, err := swiglu.Backward(l, g, x, pre)
		return post, dW, err
	}
	pCPU, wCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	pB, wB, err := run(be)
	if err != nil {
		return err
	}
	maxP, maxW := maxAbsDiff(pCPU.Data, pB.Data), maxAbsDiff(wCPU.Data, wB.Data)
	if maxP > tol || maxW > tol {
		return fmt.Errorf("%s postΔ=%g dWΔ=%g tol=%g", label, maxP, maxW, tol)
	}
	fmt.Printf("(%s postΔ=%.3g dWΔ=%.3g) ", label, maxP, maxW)
	return nil
}

func maxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var max float64
	for i := 0; i < n; i++ {
		e := math.Abs(float64(a[i] - b[i]))
		if e > max {
			max = e
		}
	}
	return max
}
