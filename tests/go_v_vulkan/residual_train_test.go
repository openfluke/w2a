package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanResidualTrain runs Go↔Vulkan Residual NormalBP (leaf + volumetric).
func TestGoVVulkanResidualTrain(t *testing.T) {
	if err := gvr.RunResidualTrain(); err != nil {
		t.Fatal(err)
	}
}
