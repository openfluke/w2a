package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppCNNTrain runs Go↔C++ CNN1/2/3 NormalBP (leaf + volumetric).
func TestGoVCppCNNTrain(t *testing.T) {
	if err := gvc.RunCNNTrain(); err != nil {
		t.Fatal(err)
	}
}
