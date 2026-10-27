package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaLayerNormTrain runs Go↔CUDA LayerNorm NormalBP (leaf + volumetric).
func TestGoVCudaLayerNormTrain(t *testing.T) {
	if err := gvc.RunLayerNormTrain(); err != nil {
		t.Fatal(err)
	}
}
