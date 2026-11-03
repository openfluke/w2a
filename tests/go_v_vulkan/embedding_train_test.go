package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanEmbeddingTrain runs Go↔Vulkan Embedding NormalBP (leaf + volumetric).
func TestGoVVulkanEmbeddingTrain(t *testing.T) {
	if err := gvr.RunEmbeddingTrain(); err != nil {
		t.Fatal(err)
	}
}
