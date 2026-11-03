package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanResidual — Go↔Vulkan Residual F(x)+x (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanResidual$'
func TestGoVVulkanResidual(t *testing.T) {
	if err := gvr.RunResidual(); err != nil {
		t.Fatal(err)
	}
}
