package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppDenseTrain runs Go↔C++ Dense NormalBP (leaf + volumetric).
func TestGoVCppDenseTrain(t *testing.T) {
	if err := gvc.RunDenseTrain(); err != nil {
		t.Fatal(err)
	}
}
