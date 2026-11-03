package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanRMSNorm — Go↔Vulkan RMSNorm FormatNone forward (exact_bit + speed).
func TestGoVVulkanRMSNorm(t *testing.T) {
	if err := gvr.RunRMSNorm(); err != nil {
		t.Fatal(err)
	}
}
