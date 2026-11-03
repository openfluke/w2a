package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanParallel — Go↔Vulkan Parallel forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanParallel$'
func TestGoVVulkanParallel(t *testing.T) {
	if err := gvr.RunParallel(); err != nil {
		t.Fatal(err)
	}
}
