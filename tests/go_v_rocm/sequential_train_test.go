package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSequentialTrain runs Go↔ROCm Sequential NormalBP (leaf + volumetric).
func TestGoVRocmSequentialTrain(t *testing.T) {
	if err := gvr.RunSequentialTrain(); err != nil {
		t.Fatal(err)
	}
}
