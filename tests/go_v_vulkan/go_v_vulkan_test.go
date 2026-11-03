package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanDenseForward — Go↔Vulkan Dense FormatNone×34 exact_bit + Go/C++/Vulkan speed.
func TestGoVVulkanDenseForward(t *testing.T) {
	if err := gvr.RunDenseForward(); err != nil {
		t.Fatal(err)
	}
}
