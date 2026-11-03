package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmResidualTrain runs Go↔ROCm Residual NormalBP (leaf + volumetric).
func TestGoVRocmResidualTrain(t *testing.T) {
	if err := gvr.RunResidualTrain(); err != nil {
		t.Fatal(err)
	}
}
