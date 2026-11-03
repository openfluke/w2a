package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSoftmax — Go↔Vulkan Softmax forward (all Kind ×34 exact_bit + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanSoftmax$'
func TestGoVVulkanSoftmax(t *testing.T) {
	if err := gvr.RunSoftmax(); err != nil {
		t.Fatal(err)
	}
}
