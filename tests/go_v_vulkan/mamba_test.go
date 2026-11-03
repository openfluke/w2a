package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanMamba — Go↔Vulkan Mamba forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanMamba$'
func TestGoVVulkanMamba(t *testing.T) {
	if err := gvr.RunMamba(); err != nil {
		t.Fatal(err)
	}
}
