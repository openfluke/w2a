package mha

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// SoftmaxSigmoid public-API smoke (Validate + Forward + Backward).
func softmaxSigmoidSmoke() error {
	cfg := mha.Config{
		DModel: 8, NumHeads: 2, HeadDim: 4, MaxSeqLen: 8,
		Mask: mha.MaskBidirectional, Pos: mha.PosNone, Mode: mha.ModeSelf,
		Softmax: mha.SoftmaxSigmoid, Causal: false,
	}
	l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	x := makeInput(1, 3, cfg.DModel)
	pre, post, err := mha.Forward(l, x)
	if err != nil {
		return fmt.Errorf("sigmoid fwd: %w", err)
	}
	g := core.NewTensor[float32](post.Shape...)
	for i := range g.Data {
		g.Data[i] = 1
	}
	if _, _, err := mha.Backward(l, g, x, pre); err != nil {
		return fmt.Errorf("sigmoid bwd: %w", err)
	}
	fmt.Printf("(SoftmaxSigmoid fwd+bwd) ")
	return nil
}

// Train-time attention Dropout: mask written, bwd succeeds, Dropout=1 rejected.
func dropoutTrainSmoke() error {
	cfg := mha.Config{
		DModel: 8, NumHeads: 2, HeadDim: 4, MaxSeqLen: 8,
		Mask: mha.MaskBidirectional, Pos: mha.PosNone, Mode: mha.ModeSelf,
		Softmax: mha.SoftmaxStandard, Causal: false, Dropout: 0.5,
	}
	bad := cfg
	bad.Dropout = 1.0
	if err := bad.Validate(); err == nil {
		return fmt.Errorf("Dropout=1 should reject")
	}
	l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	l.Training = true
	l.RNG = rand.New(rand.NewSource(7))
	x := makeInput(1, 3, cfg.DModel)
	pre, post, err := mha.Forward(l, x)
	if err != nil {
		return fmt.Errorf("dropout fwd: %w", err)
	}
	if len(l.DropMask) == 0 {
		return fmt.Errorf("DropMask empty after train forward")
	}
	var seen0, seen1 bool
	for _, b := range l.DropMask {
		if b == 0 {
			seen0 = true
		}
		if b == 1 {
			seen1 = true
		}
	}
	if !seen0 && !seen1 {
		return fmt.Errorf("DropMask had no 0/1 bytes")
	}
	g := core.NewTensor[float32](post.Shape...)
	for i := range g.Data {
		g.Data[i] = 1
	}
	if _, _, err := mha.Backward(l, g, x, pre); err != nil {
		return fmt.Errorf("dropout bwd: %w", err)
	}
	fmt.Printf("(Dropout=0.5 train mask len=%d) ", len(l.DropMask))
	return nil
}

// GPUAttnSupported gate + CPU↔WebGPU parity for decoder causal FormatNone f32.
func gpuAttnParity() error {
	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.HeadDim = 4
	cfg.MaxSeqLen = 16
	lGate, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	if !mha.GPUAttnSupported(lGate) {
		return fmt.Errorf("DecoderCausal should support GPU attn")
	}
	// Sigmoid / train-dropout must not take GPU attn path.
	sig := cfg
	sig.Softmax = mha.SoftmaxSigmoid
	lSig, err := newLayer(sig, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	if mha.GPUAttnSupported(lSig) {
		return fmt.Errorf("SoftmaxSigmoid must not GPUAttnSupported")
	}
	lDrop, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	lDrop.Cfg.Dropout = 0.3
	lDrop.Training = true
	if mha.GPUAttnSupported(lDrop) {
		return fmt.Errorf("train Dropout must not GPUAttnSupported")
	}

	if !webgpu.Available() {
		fmt.Printf("(no GPU — gate checks only) ")
		return nil
	}
	x := makeInput(1, 4, cfg.DModel)
	run := func(be core.Backend) (*core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, nil, err
		}
		pre, post, err := mha.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dW, err := mha.Backward(l, g, x, pre)
		return post, dW, err
	}
	postCPU, dWCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return fmt.Errorf("cpu: %w", err)
	}
	postGPU, dWGPU, err := run(core.BackendWebGPU)
	if err != nil {
		return fmt.Errorf("gpu: %w", err)
	}
	maxP := maxAbsDiff(postCPU.Data, postGPU.Data)
	maxW := maxAbsDiff(dWCPU.Data, dWGPU.Data)
	const tol = 2e-2
	if maxP > tol {
		return fmt.Errorf("CPU↔GPU post maxΔ=%g > %g", maxP, tol)
	}
	if maxW > tol {
		return fmt.Errorf("CPU↔GPU dW maxΔ=%g > %g", maxW, tol)
	}
	fmt.Printf("(GPU attn parity postΔ=%.3g dWΔ=%.3g) ", maxP, maxW)
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
