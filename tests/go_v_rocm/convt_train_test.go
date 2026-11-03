package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmConvTTrain runs Go↔ROCm ConvT1/2/3 NormalBP (leaf + volumetric).
func TestGoVRocmConvTTrain(t *testing.T) {
	if err := gvr.RunConvTTrain(); err != nil {
		t.Fatal(err)
	}
}
