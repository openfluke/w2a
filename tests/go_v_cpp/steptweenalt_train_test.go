package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppStepTweenAltTrain(t *testing.T) {
	if err := gvc.RunStepTweenAltTrain(); err != nil {
		t.Fatal(err)
	}
}
