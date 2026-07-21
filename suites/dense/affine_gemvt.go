package dense

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// AffinePacked WebGPU fwd+bwd vs CPU (resident GEMV + GEMVT).
func affinePackedWebGPUGEMVTParity() error {
	if !webgpu.Available() {
		fmt.Printf("(no GPU — skip) ")
		return nil
	}
	const in, out, batch = 64, 32, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}
	run := func(be core.Backend) (*core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatAffinePacked, init)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		pre, post, err := dense.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		gIn, dW, err := dense.Backward(l, g, x, pre)
		_ = gIn
		return post, dW, err
	}
	pCPU, wCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return fmt.Errorf("cpu: %w", err)
	}
	pGPU, wGPU, err := run(core.BackendWebGPU)
	if err != nil {
		return fmt.Errorf("gpu: %w", err)
	}
	const tol = 5e-2
	var maxP, maxW float64
	for i := range pCPU.Data {
		e := math.Abs(float64(pCPU.Data[i] - pGPU.Data[i]))
		if e > maxP {
			maxP = e
		}
	}
	for i := range wCPU.Data {
		e := math.Abs(float64(wCPU.Data[i] - wGPU.Data[i]))
		if e > maxW {
			maxW = e
		}
	}
	if maxP > tol || maxW > tol {
		return fmt.Errorf("Affine GEMVT WebGPU postΔ=%g dWΔ=%g", maxP, maxW)
	}
	fmt.Printf("(AffinePacked GEMVT postΔ=%.3g dWΔ=%.3g) ", maxP, maxW)
	return nil
}
