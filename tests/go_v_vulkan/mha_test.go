package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanMHA — Go↔Vulkan MHA ModeSelf forward only (exact_bit + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanMHA$'
func TestGoVVulkanMHA(t *testing.T) {
	if err := gvr.RunMHA(); err != nil {
		t.Fatal(err)
	}
}
