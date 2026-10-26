package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVMetalStepProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{parallel.ModeStepTweenSplitLinear, parallel.ModeStepTweenSplitFastProxy, parallel.ModeStepTweenSplitLinearCache, parallel.ModeStepTweenSplitHeadProxyAsync, parallel.ModeStepTweenSplitSparse} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvm.CasesStepProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
