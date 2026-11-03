package go_v_vulkan_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVVulkanStepProxyTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepTweenSplitLinear,
		parallel.ModeStepTweenSplitFastProxy,
		parallel.ModeStepTweenSplitLinearCache,
		parallel.ModeStepTweenSplitHeadProxyAsync,
		parallel.ModeStepTweenSplitSparse,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvr.CasesStepProxyTrain(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
