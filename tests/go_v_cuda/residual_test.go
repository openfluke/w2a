package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaResidual — Go↔CUDA Residual F(x)+x (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaResidual$'
func TestGoVCudaResidual(t *testing.T) {
	if err := gvc.RunResidual(); err != nil {
		t.Fatal(err)
	}
}
