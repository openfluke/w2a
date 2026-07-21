package dense

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

func repeatForwardDet() error {
	const in, out, batch = 64, 32, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}
	run := func() (*core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
		if err != nil {
			return nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		_, post, err := dense.Forward(l, x)
		return post, err
	}
	base, err := run()
	if err != nil {
		return err
	}
	var max float64
	for i := 0; i < 3; i++ {
		p, err := run()
		if err != nil {
			return err
		}
		if d := suites.MaxAbsDiff(base.Data, p.Data); d > max {
			max = d
		}
	}
	if err := suites.RequireDet("dense repeat", max, suites.DetTolFwd); err != nil {
		return err
	}
	fmt.Printf("(repeat Δ=%.3g) ", max)
	return nil
}

func scmcFwdBwdDet() error {
	const in, out, batch = 64, 32, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}
	run := func(multi bool) (post, dW []float32, err error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = multi
		l.Core.MultiCore = multi
		pre, p, err := dense.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](p.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dw, err := dense.Backward(l, g, x, pre)
		if err != nil {
			return nil, nil, err
		}
		return suites.CloneF32(p.Data), suites.CloneF32(dw.Data), nil
	}
	pSC, wSC, err := run(false)
	if err != nil {
		return err
	}
	pMC, wMC, err := run(true)
	if err != nil {
		return err
	}
	dP := suites.MaxAbsDiff(pSC, pMC)
	dW := suites.MaxAbsDiff(wSC, wMC)
	if err := suites.RequireDet("dense fwd SC↔MC", dP, suites.DetTolFwd); err != nil {
		return err
	}
	if err := suites.RequireDet("dense bwd SC↔MC", dW, suites.DetTolBwd); err != nil {
		return err
	}
	fmt.Printf("(SC↔MC postΔ=%.3g dWΔ=%.3g) ", dP, dW)
	return nil
}
