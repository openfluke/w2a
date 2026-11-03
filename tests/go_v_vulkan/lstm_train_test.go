package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanLSTMTrain runs Go↔Vulkan LSTM NormalBP (leaf + volumetric).
func TestGoVVulkanLSTMTrain(t *testing.T) {
	if err := gvr.RunLSTMTrain(); err != nil {
		t.Fatal(err)
	}
}
