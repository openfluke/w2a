package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppMHATrain runs Go↔C++ MHA NormalBP (leaf + volumetric).
func TestGoVCppMHATrain(t *testing.T) {
	if err := gvc.RunMHATrain(); err != nil {
		t.Fatal(err)
	}
}
