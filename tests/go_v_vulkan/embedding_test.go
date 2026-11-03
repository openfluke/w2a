package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanEmbedding — Go↔Vulkan Embedding gather (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanEmbedding$'
func TestGoVVulkanEmbedding(t *testing.T) {
	if err := gvr.RunEmbedding(); err != nil {
		t.Fatal(err)
	}
}
