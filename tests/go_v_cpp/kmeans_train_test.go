package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppKMeansTrain(t *testing.T) {
	if err := gvc.RunKMeansTrain(); err != nil {
		t.Fatal(err)
	}
}
