package go_v_cuda_test

import (
	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVCudaStepProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepTweenSplitLinear,
		parallel.ModeStepTweenSplitFastProxy,
		parallel.ModeStepTweenSplitLinearCache,
		parallel.ModeStepTweenSplitHeadProxyAsync,
		parallel.ModeStepTweenSplitSparse,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvc.CasesStepProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
