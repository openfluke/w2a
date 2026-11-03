package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSoftmax — Go↔ROCm Softmax forward (all Kind ×34 exact_bit + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmSoftmax$'
func TestGoVRocmSoftmax(t *testing.T) {
	if err := gvr.RunSoftmax(); err != nil {
		t.Fatal(err)
	}
}
