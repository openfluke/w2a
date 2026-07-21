package softmax

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/softmax"
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
		_, post, err := softmax.Forward(l, x)
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
	if err := suites.RequireDet("softmax repeat", max, suites.DetTolFwd); err != nil {
		return err
	}
	fmt.Printf("(repeat Δ=%.3g) ", max)
	return nil
}

func scmcFwdBwdDet() error {
	cfg := tinyCfg()
	x := makeInput(cfg, 2)
	run := func(multi bool) (post []float32, err error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
		if err != nil {
			return nil, err
		}
		l.Exec.MultiCore = multi
		l.Core.MultiCore = multi
		pre, p, err := softmax.Forward(l, x)
		if err != nil {
			return nil, err
		}
		g := core.NewTensor[float32](p.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		if _, _, err := softmax.Backward(l, g, x, pre); err != nil {
			return nil, err
		}
		return suites.CloneF32(p.Data), nil
	}
	pSC, err := run(false)
	if err != nil {
		return err
	}
	pMC, err := run(true)
	if err != nil {
		return err
	}
	dP := suites.MaxAbsDiff(pSC, pMC)
	if err := suites.RequireDet("softmax fwd SC↔MC", dP, suites.DetTolFwd); err != nil {
		return err
	}
	fmt.Printf("(SC↔MC postΔ=%.3g) ", dP)
	return nil
}
