package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppStepTweenTrain(t *testing.T) {
	if err := gvc.RunStepTweenTrain(); err != nil {
		t.Fatal(err)
	}
}
