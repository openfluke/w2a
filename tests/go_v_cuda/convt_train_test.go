package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaConvTTrain runs Go↔CUDA ConvT1/2/3 NormalBP (leaf + volumetric).
func TestGoVCudaConvTTrain(t *testing.T) {
	if err := gvc.RunConvTTrain(); err != nil {
		t.Fatal(err)
	}
}
