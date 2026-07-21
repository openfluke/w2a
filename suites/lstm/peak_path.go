package lstm

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

func fusedWebGPUParity() error {
	if !webgpu.Available() {
		fmt.Printf("(no GPU — skip) ")
		return nil
	}
	cfg := tinyCfg()
	x := makeInput(cfg, 2)
	run := func(be core.Backend) (*core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, nil, err
		}
		pre, post, err := lstm.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dW, err := lstm.Backward(l, g, x, pre)
		return post, dW, err
	}
	pCPU, wCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	pGPU, wGPU, err := run(core.BackendWebGPU)
	if err != nil {
		return err
	}
	maxP, maxW := maxAbsDiff(pCPU.Data, pGPU.Data), maxAbsDiff(wCPU.Data, wGPU.Data)
	const tol = 5e-2
	if maxP > tol || maxW > tol {
		return fmt.Errorf("LSTM fused WebGPU postΔ=%g dWΔ=%g", maxP, maxW)
	}
	fmt.Printf("(fused FormatNone f32 postΔ=%.3g dWΔ=%.3g) ", maxP, maxW)
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
