package embedding

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/quant"
)

func repeatForwardDet() error {
	cfg := tinyCfg()
	x := makeInput(cfg, 2)
	run := func() (*core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
		if err != nil {
			return nil, err
		}
		l.Exec.MultiCore = false
		_, post, err := embedding.Forward(l, x)
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
	if err := suites.RequireDet("embedding repeat", max, suites.DetTolFwd); err != nil {
		return err
	}
	fmt.Printf("(repeat Δ=%.3g) ", max)
	return nil
}

func scmcFwdBwdDet() error {
	cfg := tinyCfg()
	x := makeInput(cfg, 2)
	run := func(multi bool) (post, dW []float32, err error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.MultiCore = multi
		l.Core.MultiCore = multi
		pre, p, err := embedding.Forward(l, x)
		if err != nil {
			return nil, nil, err
		}
		g := core.NewTensor[float32](p.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dw, err := embedding.Backward(l, g, x, pre)
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
	dP, dW := suites.MaxAbsDiff(pSC, pMC), suites.MaxAbsDiff(wSC, wMC)
	if err := suites.RequireDet("embedding fwd SC↔MC", dP, suites.DetTolFwd); err != nil {
		return err
	}
	if err := suites.RequireDet("embedding bwd SC↔MC", dW, suites.DetTolBwd); err != nil {
		return err
	}
	fmt.Printf("(SC↔MC postΔ=%.3g dWΔ=%.3g) ", dP, dW)
	return nil
}
