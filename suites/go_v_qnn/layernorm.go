package go_v_qnn

import (
	"fmt"
	"math"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func caseLayerNorm() error {
	batch, dim := 8, 64
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	x := make([]float32, batch*dim)
	for i := range gamma {
		gamma[i] = 1.0 + float32(i%3)*0.01
		beta[i] = float32(i%4) * 0.005
	}
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.019)) * 0.4
	}
	cfg := layernorm.Config{Dim: dim, Eps: 1e-5}

	mk := func(b core.Backend) (*layernorm.Layer, *core.Tensor[float32], error) {
		l, err := layernorm.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, gamma, beta)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](batch, dim)
		copy(t.Data, x)
		return l, t, nil
	}

	fmt.Println("      protocol: SyncToNative → SyncToQNN → time Forwards (QNN online-graph for norms)")

	goL, goX, err := mk(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	_, goPost, err := layernorm.Forward(goL, goX)
	if err != nil {
		return err
	}
	goOut := append([]float32(nil), goPost.Data...)
	goUS, err := benchUS(func() error {
		_, _, e := layernorm.Forward(goL, goX)
		return e
	}, 500)
	if err != nil {
		return err
	}
	t := tripleTiming{goUS: goUS}

	if nativecpp.NativeCPUAvailable() {
		natL, natX, err := mk(core.BackendNativeCPU)
		if err != nil {
			return err
		}
		if err := natL.SyncToNative(); err != nil {
			return err
		}
		_, natPost, err := layernorm.Forward(natL, natX)
		if err != nil {
			return fmt.Errorf("native: %w", err)
		}
		t.goVsNat = suites.CompareF32Spectrum(goOut, natPost.Data)
		if err := gateNative(t.goVsNat); err != nil {
			return err
		}
		t.natUS, err = benchUS(func() error {
			_, _, e := layernorm.Forward(natL, natX)
			return e
		}, 500)
		if err != nil {
			return err
		}
		t.natRan = true
	}

	qL, qX, err := mk(core.BackendQNN)
	if err != nil {
		return err
	}
	if err := qL.SyncToQNN(); err != nil {
		return err
	}
	_, qPost, err := layernorm.Forward(qL, qX)
	if err != nil {
		return fmt.Errorf("qnn: %w", err)
	}
	t.goVsQNN = suites.CompareF32Spectrum(goOut, qPost.Data)
	if err := gateQNN(t.goVsQNN); err != nil {
		return err
	}
	t.qnnUS, err = benchUS(func() error {
		_, _, e := layernorm.Forward(qL, qX)
		return e
	}, 4)
	if err != nil {
		return err
	}
	t.qnnRan = true
	printTriple("LayerNorm [8,64]", t)
	return nil
}
