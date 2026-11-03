package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanRNN — Go↔Vulkan RNN FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanRNN$'
func TestGoVVulkanRNN(t *testing.T) {
	if err := gvr.RunRNN(); err != nil {
		t.Fatal(err)
	}
}
