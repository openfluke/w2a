package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanMHATrain runs Go↔Vulkan MHA NormalBP (leaf + volumetric).
func TestGoVVulkanMHATrain(t *testing.T) {
	if err := gvr.RunMHATrain(); err != nil {
		t.Fatal(err)
	}
}
