package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaDenseForward — Go↔CUDA Dense FormatNone×34 exact_bit + Go/C++/CUDA speed.
func TestGoVCudaDenseForward(t *testing.T) {
	if err := gvc.RunDenseForward(); err != nil {
		t.Fatal(err)
	}
}
