package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSoftmaxTrain runs Go↔C++ Softmax NormalBP (leaf all Kind + volumetric).
func TestGoVCppSoftmaxTrain(t *testing.T) {
	if err := gvc.RunSoftmaxTrain(); err != nil {
		t.Fatal(err)
	}
}
