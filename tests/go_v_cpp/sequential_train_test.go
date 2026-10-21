package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppSequentialTrain(t *testing.T) {
	if err := gvc.RunSequentialTrain(); err != nil {
		t.Fatal(err)
	}
}
