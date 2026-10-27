package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaLSTMTrain runs Go↔CUDA LSTM NormalBP (leaf + volumetric).
func TestGoVCudaLSTMTrain(t *testing.T) {
	if err := gvc.RunLSTMTrain(); err != nil {
		t.Fatal(err)
	}
}
