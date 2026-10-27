package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSwiGLU — Go↔CUDA SwiGLU forward only (exact_bit + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaSwiGLU$'
func TestGoVCudaSwiGLU(t *testing.T) {
	if err := gvc.RunSwiGLU(); err != nil {
		t.Fatal(err)
	}
}
