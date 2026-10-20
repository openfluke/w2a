package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSwiGLU runs only Go↔C++ SwiGLU cases (use: go test -run '^TestGoVCppSwiGLU$').
func TestGoVCppSwiGLU(t *testing.T) {
	if err := gvc.RunSwiGLU(); err != nil {
		t.Fatal(err)
	}
}
