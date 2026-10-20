package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSoftmax runs Go↔C++ Softmax all Kind variants.
func TestGoVCppSoftmax(t *testing.T) {
	if err := gvc.RunSoftmax(); err != nil {
		t.Fatal(err)
	}
}
