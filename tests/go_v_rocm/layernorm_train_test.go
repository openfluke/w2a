package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmLayerNormTrain runs Go↔ROCm LayerNorm NormalBP (leaf + volumetric).
func TestGoVRocmLayerNormTrain(t *testing.T) {
	if err := gvr.RunLayerNormTrain(); err != nil {
		t.Fatal(err)
	}
}
