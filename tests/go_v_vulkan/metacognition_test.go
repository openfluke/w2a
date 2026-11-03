package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanMetacognition — Go↔Vulkan Metacognition forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanMetacognition$'
func TestGoVVulkanMetacognition(t *testing.T) {
	if err := gvr.RunMetacognition(); err != nil {
		t.Fatal(err)
	}
}
