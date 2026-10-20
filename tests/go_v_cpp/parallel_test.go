package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppParallel runs Go↔C++ Parallel Dense combine cases (7 modes + Add×34).
func TestGoVCppParallel(t *testing.T) {
	if err := gvc.RunParallel(); err != nil {
		t.Fatal(err)
	}
}
