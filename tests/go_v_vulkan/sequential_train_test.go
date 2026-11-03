package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSequentialTrain runs Go↔Vulkan Sequential NormalBP (leaf + volumetric).
func TestGoVVulkanSequentialTrain(t *testing.T) {
	if err := gvr.RunSequentialTrain(); err != nil {
		t.Fatal(err)
	}
}
