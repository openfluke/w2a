package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppMHA runs only Go↔C++ MHA cases (use: go test -run '^TestGoVCppMHA$').
func TestGoVCppMHA(t *testing.T) {
	if err := gvc.RunMHA(); err != nil {
		t.Fatal(err)
	}
}
