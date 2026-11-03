package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanLayerNormTrain runs Go↔Vulkan LayerNorm NormalBP (leaf + volumetric).
func TestGoVVulkanLayerNormTrain(t *testing.T) {
	if err := gvr.RunLayerNormTrain(); err != nil {
		t.Fatal(err)
	}
}
