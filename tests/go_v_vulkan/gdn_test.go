package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanGDN — Go↔Vulkan GDN forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanGDN$'
func TestGoVVulkanGDN(t *testing.T) {
	if err := gvr.RunGDN(); err != nil {
		t.Fatal(err)
	}
}
