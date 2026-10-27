package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaRNN — Go↔CUDA RNN FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaRNN$'
func TestGoVCudaRNN(t *testing.T) {
	if err := gvc.RunRNN(); err != nil {
		t.Fatal(err)
	}
}
