package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSequentialTrain runs Go↔CUDA Sequential NormalBP (leaf + volumetric).
func TestGoVCudaSequentialTrain(t *testing.T) {
	if err := gvc.RunSequentialTrain(); err != nil {
		t.Fatal(err)
	}
}
