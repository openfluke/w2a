package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaLSTM — Go↔CUDA LSTM FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaLSTM$'
func TestGoVCudaLSTM(t *testing.T) {
	if err := gvc.RunLSTM(); err != nil {
		t.Fatal(err)
	}
}
