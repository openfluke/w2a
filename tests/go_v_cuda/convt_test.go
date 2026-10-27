package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaConvT — Go↔CUDA ConvT1/2/3 FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaConvT$'
func TestGoVCudaConvT(t *testing.T) {
	if err := gvc.RunConvT(); err != nil {
		t.Fatal(err)
	}
}
