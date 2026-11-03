package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanCNNTrain runs Go↔Vulkan CNN1/2/3 NormalBP (leaf + volumetric).
func TestGoVVulkanCNNTrain(t *testing.T) {
	if err := gvr.RunCNNTrain(); err != nil {
		t.Fatal(err)
	}
}
