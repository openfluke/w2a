package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSwiGLUTrain runs Go↔CUDA SwiGLU NormalBP (leaf + volumetric).
func TestGoVCudaSwiGLUTrain(t *testing.T) {
	if err := gvc.RunSwiGLUTrain(); err != nil {
		t.Fatal(err)
	}
}
