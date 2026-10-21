package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppTweenSplitLinearTrain(t *testing.T) {
	if err := gvc.RunTweenSplitLinearTrain(); err != nil {
		t.Fatal(err)
	}
}
