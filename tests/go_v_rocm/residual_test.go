package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmResidual — Go↔ROCm Residual F(x)+x (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmResidual$'
func TestGoVRocmResidual(t *testing.T) {
	if err := gvr.RunResidual(); err != nil {
		t.Fatal(err)
	}
}
