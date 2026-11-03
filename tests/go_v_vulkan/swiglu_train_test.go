package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSwiGLUTrain runs Go↔Vulkan SwiGLU NormalBP (leaf + volumetric).
func TestGoVVulkanSwiGLUTrain(t *testing.T) {
	if err := gvr.RunSwiGLUTrain(); err != nil {
		t.Fatal(err)
	}
}
