package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppRMSNorm runs only Go↔C++ RMSNorm cases (use: go test -run '^TestGoVCppRMSNorm$').
func TestGoVCppRMSNorm(t *testing.T) {
	if err := gvc.RunRMSNorm(); err != nil {
		t.Fatal(err)
	}
}
