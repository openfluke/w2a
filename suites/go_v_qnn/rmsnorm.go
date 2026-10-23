package go_v_qnn

import (
	"fmt"
	"math"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func caseRMSNorm() error {
	batch, dim := 8, 64
	gamma := make([]float32, dim)
	x := make([]float32, batch*dim)
	for i := range gamma {
		gamma[i] = 0.9 + float32(i%5)*0.02
	}
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.021)) * 0.5
	}
	cfg := rmsnorm.Config{Dim: dim, Eps: 1e-6}

	mk := func(b core.Backend) (*rmsnorm.Layer, *core.Tensor[float32], error) {
		l, err := rmsnorm.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, gamma)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](batch, dim)
		copy(t.Data, x)
		return l, t, nil
	}

	fmt.Println("      protocol: SyncToNative → SyncToQNN → time Forwards (QNN still online-graph for norms)")

	goL, goX, err := mk(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	_, goPost, err := rmsnorm.Forward(goL, goX)
	if err != nil {
		return err
	}
	goOut := append([]float32(nil), goPost.Data...)
	goUS, err := benchUS(func() error {
		_, _, e := rmsnorm.Forward(goL, goX)
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
		_, natPost, err := rmsnorm.Forward(natL, natX)
		if err != nil {
			return fmt.Errorf("native: %w", err)
		}
		t.goVsNat = suites.CompareF32Spectrum(goOut, natPost.Data)
		if err := gateNative(t.goVsNat); err != nil {
			return err
		}
		t.natUS, err = benchUS(func() error {
			_, _, e := rmsnorm.Forward(natL, natX)
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
	_, qPost, err := rmsnorm.Forward(qL, qX)
	if err != nil {
		return fmt.Errorf("qnn: %w", err)
	}
	t.goVsQNN = suites.CompareF32Spectrum(goOut, qPost.Data)
	if err := gateQNN(t.goVsQNN); err != nil {
		return err
	}
	t.qnnUS, err = benchUS(func() error {
		_, _, e := rmsnorm.Forward(qL, qX)
		return e
	}, 4)
	if err != nil {
		return err
	}
	t.qnnRan = true
	t.qnnPrepared = false
	printTriple("RMSNorm [8,64]", t)
	return nil
}
