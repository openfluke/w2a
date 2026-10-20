package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppResidual runs Go↔C++ Residual Dense F(x)+x cases.
func TestGoVCppResidual(t *testing.T) {
	if err := gvc.RunResidual(); err != nil {
		t.Fatal(err)
	}
}
