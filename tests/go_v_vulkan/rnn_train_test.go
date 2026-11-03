package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanRNNTrain runs Go↔Vulkan RNN NormalBP (leaf + volumetric).
func TestGoVVulkanRNNTrain(t *testing.T) {
	if err := gvr.RunRNNTrain(); err != nil {
		t.Fatal(err)
	}
}
