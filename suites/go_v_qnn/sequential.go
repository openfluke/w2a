package go_v_qnn

import (
	"fmt"
	"math"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func caseSequentialBig() error {
	batch, dim, depth := 8, 128, 3
	cfg := sequential.Config{Dim: dim, Depth: depth}
	n := depth * dim * dim
	packed := make([]float32, n)
	x := make([]float32, batch*dim)
	for i := range packed {
		packed[i] = float32(math.Sin(float64(i)*0.011)+0.05) * 0.03
	}
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.029)) * 0.2
	}

	mk := func(b core.Backend) (*sequential.Layer, *core.Tensor[float32], error) {
		l, err := sequential.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, packed)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		l.Core.MultiCore = false
		for _, ch := range l.Children {
			if ch == nil {
				continue
			}
			ch.Exec.Backend = b
			ch.Exec.MultiCore = false
			ch.Core.MultiCore = false
		}
		t := core.NewTensor[float32](batch, dim)
		copy(t.Data, x)
		return l, t, nil
	}

	fmt.Println("      protocol: SyncToNative → SyncToQNN → time Forwards (sequence test; QNN online per child)")

	goL, goX, err := mk(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	_, goPost, err := sequential.Forward(goL, goX)
	if err != nil {
		return err
	}
	goOut := append([]float32(nil), goPost.Data...)
	goUS, err := benchUS(func() error {
		_, _, e := sequential.Forward(goL, goX)
		return e
	}, 200)
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
		_, natPost, err := sequential.Forward(natL, natX)
		if err != nil {
			return fmt.Errorf("native: %w", err)
		}
		t.goVsNat = suites.CompareF32Spectrum(goOut, natPost.Data)
		if err := gateNative(t.goVsNat); err != nil {
			return err
		}
		t.natUS, err = benchUS(func() error {
			_, _, e := sequential.Forward(natL, natX)
			return e
		}, 200)
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
		return fmt.Errorf("SyncToQNN: %w", err)
	}
	_, qPost, err := sequential.Forward(qL, qX)
	if err != nil {
		return fmt.Errorf("qnn: %w", err)
	}
	t.goVsQNN = suites.CompareF32Spectrum(goOut, qPost.Data)
	if err := gateQNN(t.goVsQNN); err != nil {
		return err
	}
	t.qnnUS, err = benchUS(func() error {
		_, _, e := sequential.Forward(qL, qX)
		return e
	}, 3)
	if err != nil {
		return err
	}
	t.qnnRan = true
	printTriple(fmt.Sprintf("Sequential dim=%d depth=%d", dim, depth), t)
	return nil
}
