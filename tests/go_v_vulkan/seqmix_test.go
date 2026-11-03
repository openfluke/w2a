package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// TestGoVVulkanSeqmix — Go↔Vulkan Seqmix forward parity.
// Run: go test ./tests/go_v_vulkan/ -v -run '^TestGoVVulkanSeqmix$'
func TestGoVVulkanSeqmix(t *testing.T) {
	if err := gvr.RunSeqmix(); err != nil {
		t.Fatal(err)
	}
}
