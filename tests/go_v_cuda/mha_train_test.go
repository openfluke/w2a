package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaMHATrain runs Go↔CUDA MHA NormalBP (leaf + volumetric).
func TestGoVCudaMHATrain(t *testing.T) {
	if err := gvc.RunMHATrain(); err != nil {
		t.Fatal(err)
	}
}
