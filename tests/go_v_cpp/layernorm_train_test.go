package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppLayerNormTrain runs Go↔C++ LayerNorm NormalBP (leaf + volumetric).
func TestGoVCppLayerNormTrain(t *testing.T) {
	if err := gvc.RunLayerNormTrain(); err != nil {
		t.Fatal(err)
	}
}
