package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanDenseTrain runs Go↔Vulkan Dense NormalBP (leaf + volumetric).
func TestGoVVulkanDenseTrain(t *testing.T) {
	if err := gvr.RunDenseTrain(); err != nil {
		t.Fatal(err)
	}
}
