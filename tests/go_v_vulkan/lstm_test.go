package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanLSTM — Go↔Vulkan LSTM FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanLSTM$'
func TestGoVVulkanLSTM(t *testing.T) {
	if err := gvr.RunLSTM(); err != nil {
		t.Fatal(err)
	}
}
