package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmStackTrain runs Go↔ROCm Stack NormalBP (leaf + volumetric).
func TestGoVRocmStackTrain(t *testing.T) {
	if err := gvr.RunStackTrain(); err != nil {
		t.Fatal(err)
	}
}
