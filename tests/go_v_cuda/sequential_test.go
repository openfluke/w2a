package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSequential — Go↔CUDA Sequential Dense chain (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaSequential$'
func TestGoVCudaSequential(t *testing.T) {
	if err := gvc.RunSequential(); err != nil {
		t.Fatal(err)
	}
}
