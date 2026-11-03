package go_v_rocm_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVRocmProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{
		parallel.ModeTweenSplitHeadProxy,
		parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync,
		parallel.ModeTweenSplitSparse,
		parallel.ModeStepTweenSplitHeadProxy,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvr.CasesProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestGoVRocmProxyVolumeTrain(t *testing.T) {
	for _, m := range []parallel.TrainMode{
		parallel.ModeTweenSplitHeadProxy,
		parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync,
		parallel.ModeTweenSplitSparse,
	} {
		t.Run(m.String(), func(t *testing.T) {
			for _, c := range gvr.CasesProxyVolumes(m) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
