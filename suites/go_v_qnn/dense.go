package go_v_qnn

import (
	"fmt"
	"math"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func caseDense032() error { return runDenseTriple(8, 32, 32) }
func caseDense128() error { return runDenseTriple(8, 128, 128) }

func runDenseTriple(batch, in, out int) error {
	w := make([]float32, out*in)
	bias := make([]float64, out)
	x := make([]float32, batch*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.017)+0.1) * 0.05
	}
	for i := range bias {
		bias[i] = float64(i%7) * 0.01
	}
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.013)) * 0.1
	}

	mk := func(b core.Backend) (*dense.Layer, *core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, nil, err
		}
		l.Weights.Bias = append([]float64(nil), bias...)
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		l.Core.MultiCore = false
		t := core.NewTensor[float32](batch, in)
		copy(t.Data, x)
		return l, t, nil
	}

	fmt.Println("      protocol: build → SyncToNative → SyncToQNN+PrepareQNN → time Forwards only")

	// --- Go reference (no sync) ---
	goL, goX, err := mk(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	_, goPost, err := dense.Forward(goL, goX)
	if err != nil {
		return fmt.Errorf("go forward: %w", err)
	}
	goOut := append([]float32(nil), goPost.Data...)
	goUS, err := benchUS(func() error {
		_, _, e := dense.Forward(goL, goX)
		return e
	}, 200)
	if err != nil {
		return err
	}
	t := tripleTiming{goUS: goUS}

	// --- Native: push weights, then time exec ---
	if nativecpp.NativeCPUAvailable() {
		natL, natX, err := mk(core.BackendNativeCPU)
		if err != nil {
			return err
		}
		if err := natL.SyncToNative(); err != nil {
			return fmt.Errorf("SyncToNative: %w", err)
		}
		_, natPost, err := dense.Forward(natL, natX)
		if err != nil {
			return fmt.Errorf("native forward: %w", err)
		}
		t.goVsNat = suites.CompareF32Spectrum(goOut, natPost.Data)
		if err := gateNative(t.goVsNat); err != nil {
			return err
		}
		t.natUS, err = benchUS(func() error {
			_, _, e := dense.Forward(natL, natX)
			return e
		}, 200)
		if err != nil {
			return err
		}
		t.natRan = true
	}

	// --- QNN: push weights + prepare graph, then time execute-only ---
	qL, qX, err := mk(core.BackendQNN)
	if err != nil {
		return err
	}
	if err := qL.SyncToQNN(); err != nil {
		return fmt.Errorf("SyncToQNN: %w", err)
	}
	if err := qL.PrepareQNN(batch); err != nil {
		return fmt.Errorf("PrepareQNN: %w", err)
	}
	_, qPost, err := dense.Forward(qL, qX)
	if err != nil {
		return fmt.Errorf("qnn forward: %w", err)
	}
	t.goVsQNN = suites.CompareF32Spectrum(goOut, qPost.Data)
	if err := gateQNN(t.goVsQNN); err != nil {
		return err
	}
	t.qnnUS, err = benchUS(func() error {
		_, _, e := dense.Forward(qL, qX)
		return e
	}, 50)
	if err != nil {
		return err
	}
	t.qnnRan = true
	t.qnnPrepared = true

	printTriple(fmt.Sprintf("Dense %dx%dx%d", batch, in, out), t)
	return nil
}
