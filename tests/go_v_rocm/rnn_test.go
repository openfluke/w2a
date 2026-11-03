package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmRNN — Go↔ROCm RNN FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmRNN$'
func TestGoVRocmRNN(t *testing.T) {
	if err := gvr.RunRNN(); err != nil {
		t.Fatal(err)
	}
}
