package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppRMSNormTrain runs Go↔C++ RMSNorm NormalBP (leaf + volumetric).
func TestGoVCppRMSNormTrain(t *testing.T) {
	if err := gvc.RunRMSNormTrain(); err != nil {
		t.Fatal(err)
	}
}
