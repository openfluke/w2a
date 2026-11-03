package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmMHA — Go↔ROCm MHA ModeSelf forward only (exact_bit + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmMHA$'
func TestGoVRocmMHA(t *testing.T) {
	if err := gvr.RunMHA(); err != nil {
		t.Fatal(err)
	}
}
