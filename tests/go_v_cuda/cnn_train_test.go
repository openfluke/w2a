package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaCNNTrain runs Go↔CUDA CNN1/2/3 NormalBP (leaf + volumetric).
func TestGoVCudaCNNTrain(t *testing.T) {
	if err := gvc.RunCNNTrain(); err != nil {
		t.Fatal(err)
	}
}
