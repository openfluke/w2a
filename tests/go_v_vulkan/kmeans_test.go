package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanKMeans — Go↔Vulkan KMeans forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanKMeans$'
func TestGoVVulkanKMeans(t *testing.T) {
	if err := gvr.RunKMeans(); err != nil {
		t.Fatal(err)
	}
}
