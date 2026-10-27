package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaMHA — Go↔CUDA MHA ModeSelf forward only (exact_bit + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaMHA$'
func TestGoVCudaMHA(t *testing.T) {
	if err := gvc.RunMHA(); err != nil {
		t.Fatal(err)
	}
}
