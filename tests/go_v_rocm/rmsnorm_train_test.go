package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmRMSNormTrain runs Go↔ROCm RMSNorm NormalBP (leaf + volumetric).
func TestGoVRocmRMSNormTrain(t *testing.T) {
	if err := gvr.RunRMSNormTrain(); err != nil {
		t.Fatal(err)
	}
}
