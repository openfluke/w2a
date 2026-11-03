package go_v_vulkan_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
	"testing"
)

func TestGoVVulkanMeshTweenChainTrain(t *testing.T) {
	for _, c := range gvr.CasesMeshTweenChainTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
