package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppConvTTrain runs Go↔C++ ConvT1/2/3 NormalBP (leaf + volumetric).
func TestGoVCppConvTTrain(t *testing.T) {
	if err := gvc.RunConvTTrain(); err != nil {
		t.Fatal(err)
	}
}
