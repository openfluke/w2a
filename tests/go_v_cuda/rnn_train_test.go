package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaRNNTrain runs Go↔CUDA RNN NormalBP (leaf + volumetric).
func TestGoVCudaRNNTrain(t *testing.T) {
	if err := gvc.RunRNNTrain(); err != nil {
		t.Fatal(err)
	}
}
