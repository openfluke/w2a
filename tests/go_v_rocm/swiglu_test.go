package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSwiGLU — Go↔ROCm SwiGLU forward only (exact_bit + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmSwiGLU$'
func TestGoVRocmSwiGLU(t *testing.T) {
	if err := gvr.RunSwiGLU(); err != nil {
		t.Fatal(err)
	}
}
