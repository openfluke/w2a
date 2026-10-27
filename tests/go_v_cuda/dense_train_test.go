package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaDenseTrain runs Go↔CUDA Dense NormalBP (leaf + volumetric).
func TestGoVCudaDenseTrain(t *testing.T) {
	if err := gvc.RunDenseTrain(); err != nil {
		t.Fatal(err)
	}
}
