package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanConvTTrain runs Go↔Vulkan ConvT1/2/3 NormalBP (leaf + volumetric).
func TestGoVVulkanConvTTrain(t *testing.T) {
	if err := gvr.RunConvTTrain(); err != nil {
		t.Fatal(err)
	}
}
