package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSwiGLU — Go↔Vulkan SwiGLU forward only (exact_bit + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanSwiGLU$'
func TestGoVVulkanSwiGLU(t *testing.T) {
	if err := gvr.RunSwiGLU(); err != nil {
		t.Fatal(err)
	}
}
