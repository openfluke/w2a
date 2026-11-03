package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanRMSNormTrain runs Go↔Vulkan RMSNorm NormalBP (leaf + volumetric).
func TestGoVVulkanRMSNormTrain(t *testing.T) {
	if err := gvr.RunRMSNormTrain(); err != nil {
		t.Fatal(err)
	}
}
