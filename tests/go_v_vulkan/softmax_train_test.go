package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSoftmaxTrain runs Go↔Vulkan Softmax NormalBP (leaf + volumetric).
func TestGoVVulkanSoftmaxTrain(t *testing.T) {
	if err := gvr.RunSoftmaxTrain(); err != nil {
		t.Fatal(err)
	}
}
