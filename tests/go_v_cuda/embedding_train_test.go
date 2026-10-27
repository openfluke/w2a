package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaEmbeddingTrain runs Go↔CUDA Embedding NormalBP (leaf + volumetric).
func TestGoVCudaEmbeddingTrain(t *testing.T) {
	if err := gvc.RunEmbeddingTrain(); err != nil {
		t.Fatal(err)
	}
}
