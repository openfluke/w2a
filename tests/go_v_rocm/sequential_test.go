package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSequential — Go↔ROCm Sequential Dense chain (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmSequential$'
func TestGoVRocmSequential(t *testing.T) {
	if err := gvr.RunSequential(); err != nil {
		t.Fatal(err)
	}
}
