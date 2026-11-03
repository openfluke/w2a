package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSwiGLUTrain runs Go↔ROCm SwiGLU NormalBP (leaf + volumetric).
func TestGoVRocmSwiGLUTrain(t *testing.T) {
	if err := gvr.RunSwiGLUTrain(); err != nil {
		t.Fatal(err)
	}
}
