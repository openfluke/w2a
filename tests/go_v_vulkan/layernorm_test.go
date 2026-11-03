package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanLayerNorm — Go↔Vulkan LayerNorm FormatNone forward (exact_bit + speed).
func TestGoVVulkanLayerNorm(t *testing.T) {
	if err := gvr.RunLayerNorm(); err != nil {
		t.Fatal(err)
	}
}
