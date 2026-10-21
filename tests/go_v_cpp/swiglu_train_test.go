package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSwiGLUTrain runs Go↔C++ SwiGLU NormalBP (leaf + volumetric).
func TestGoVCppSwiGLUTrain(t *testing.T) {
	if err := gvc.RunSwiGLUTrain(); err != nil {
		t.Fatal(err)
	}
}
