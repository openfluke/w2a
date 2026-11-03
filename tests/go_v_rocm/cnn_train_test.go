package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmCNNTrain runs Go↔ROCm CNN1/2/3 NormalBP (leaf + volumetric).
func TestGoVRocmCNNTrain(t *testing.T) {
	if err := gvr.RunCNNTrain(); err != nil {
		t.Fatal(err)
	}
}
