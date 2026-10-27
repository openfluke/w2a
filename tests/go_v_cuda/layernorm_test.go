package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaLayerNorm — Go↔CUDA LayerNorm FormatNone forward (exact_bit + speed).
func TestGoVCudaLayerNorm(t *testing.T) {
	if err := gvc.RunLayerNorm(); err != nil {
		t.Fatal(err)
	}
}
