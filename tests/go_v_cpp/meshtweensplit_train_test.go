package go_v_cpp_test

import (
	"testing"

	"github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppMeshTweenSplitTrain(t *testing.T) {
	if err := go_v_cpp.RunMeshTweenSplitTrain(); err != nil {
		t.Fatal(err)
	}
}
