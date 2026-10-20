package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCpp runs the full Go↔C++ suite (dense + architecture + MHA + SwiGLU + RMSNorm + LayerNorm).
func TestGoVCpp(t *testing.T) {
	if err := gvc.RunAll(); err != nil {
		t.Fatal(err)
	}
}
