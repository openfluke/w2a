package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmDenseForward — Go↔ROCm Dense FormatNone×34 exact_bit + Go/C++/ROCm speed.
func TestGoVRocmDenseForward(t *testing.T) {
	if err := gvr.RunDenseForward(); err != nil {
		t.Fatal(err)
	}
}
