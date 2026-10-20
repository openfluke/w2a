package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppConvT runs only Go↔C++ ConvT1/2/3 cases (use: go test -run '^TestGoVCppConvT$').
func TestGoVCppConvT(t *testing.T) {
	if err := gvc.RunConvT(); err != nil {
		t.Fatal(err)
	}
}
