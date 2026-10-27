package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaStackTrain runs Go↔CUDA Stack NormalBP (leaf + volumetric).
func TestGoVCudaStackTrain(t *testing.T) {
	if err := gvc.RunStackTrain(); err != nil {
		t.Fatal(err)
	}
}
