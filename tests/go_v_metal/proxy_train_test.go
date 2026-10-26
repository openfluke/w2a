package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVMetalProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{parallel.ModeTweenSplitHeadProxy, parallel.ModeTweenSplitLinear, parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinearCache, parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse, parallel.ModeStepTweenSplitHeadProxy} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvm.CasesProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestGoVMetalProxyVolumeTrain(t *testing.T) {
	for _, m := range []parallel.TrainMode{parallel.ModeTweenSplitHeadProxy, parallel.ModeTweenSplitLinear, parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinearCache, parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse} {
		t.Run(m.String(), func(t *testing.T) {
			for _, c := range gvm.CasesProxyVolumes(m) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
