package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSequential runs Go↔C++ Sequential Dense-chain cases.
func TestGoVCppSequential(t *testing.T) {
	if err := gvc.RunSequential(); err != nil {
		t.Fatal(err)
	}
}
