package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaRMSNormTrain runs Go↔CUDA RMSNorm NormalBP (leaf + volumetric).
func TestGoVCudaRMSNormTrain(t *testing.T) {
	if err := gvc.RunRMSNormTrain(); err != nil {
		t.Fatal(err)
	}
}
