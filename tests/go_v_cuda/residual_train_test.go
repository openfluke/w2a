package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaResidualTrain runs Go↔CUDA Residual NormalBP (leaf + volumetric).
func TestGoVCudaResidualTrain(t *testing.T) {
	if err := gvc.RunResidualTrain(); err != nil {
		t.Fatal(err)
	}
}
