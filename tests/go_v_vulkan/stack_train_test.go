package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanStackTrain runs Go↔Vulkan Stack NormalBP (leaf + volumetric).
func TestGoVVulkanStackTrain(t *testing.T) {
	if err := gvr.RunStackTrain(); err != nil {
		t.Fatal(err)
	}
}
