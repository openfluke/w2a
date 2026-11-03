package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSequential — Go↔Vulkan Sequential Dense chain (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanSequential$'
func TestGoVVulkanSequential(t *testing.T) {
	if err := gvr.RunSequential(); err != nil {
		t.Fatal(err)
	}
}
