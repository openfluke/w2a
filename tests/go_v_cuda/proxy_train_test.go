package go_v_cuda_test

import (
	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVCudaProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{
		parallel.ModeTweenSplitHeadProxy,
		parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync,
		parallel.ModeTweenSplitSparse,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvc.CasesProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestGoVCudaProxyVolumeTrain(t *testing.T) {
	for _, m := range []parallel.TrainMode{
		parallel.ModeTweenSplitHeadProxy,
		parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy,
		parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync,
		parallel.ModeTweenSplitSparse,
	} {
		t.Run(m.String(), func(t *testing.T) {
			for _, c := range gvc.CasesProxyVolumes(m) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
