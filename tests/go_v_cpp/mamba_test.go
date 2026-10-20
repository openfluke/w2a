package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppMamba runs Go↔C++ Mamba minimal SSM cases.
func TestGoVCppMamba(t *testing.T) {
	if err := gvc.RunMamba(); err != nil {
		t.Fatal(err)
	}
}
