package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanCNN — Go↔Vulkan CNN1/2/3 FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanCNN$'
func TestGoVVulkanCNN(t *testing.T) {
	if err := gvr.RunCNN(); err != nil {
		t.Fatal(err)
	}
}
