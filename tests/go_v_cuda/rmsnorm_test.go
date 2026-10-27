package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaRMSNorm — Go↔CUDA RMSNorm FormatNone forward (exact_bit + speed).
func TestGoVCudaRMSNorm(t *testing.T) {
	if err := gvc.RunRMSNorm(); err != nil {
		t.Fatal(err)
	}
}
