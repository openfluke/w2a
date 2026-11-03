package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanConvT — Go↔Vulkan ConvT1/2/3 FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanConvT$'
func TestGoVVulkanConvT(t *testing.T) {
	if err := gvr.RunConvT(); err != nil {
		t.Fatal(err)
	}
}
