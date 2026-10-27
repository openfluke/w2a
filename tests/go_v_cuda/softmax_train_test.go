package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSoftmaxTrain runs Go↔CUDA Softmax NormalBP (leaf + volumetric).
func TestGoVCudaSoftmaxTrain(t *testing.T) {
	if err := gvc.RunSoftmaxTrain(); err != nil {
		t.Fatal(err)
	}
}
