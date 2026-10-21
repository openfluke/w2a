package go_v_cpp_test

import (
	"testing"

	"github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppStepTweenSplitLinearTrain(t *testing.T) {
	if err := go_v_cpp.RunStepTweenSplitLinearTrain(); err != nil {
		t.Fatal(err)
	}
}
