package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmRNNTrain runs Go↔ROCm RNN NormalBP (leaf + volumetric).
func TestGoVRocmRNNTrain(t *testing.T) {
	if err := gvr.RunRNNTrain(); err != nil {
		t.Fatal(err)
	}
}
