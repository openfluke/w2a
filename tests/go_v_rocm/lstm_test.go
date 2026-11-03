package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmLSTM — Go↔ROCm LSTM FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmLSTM$'
func TestGoVRocmLSTM(t *testing.T) {
	if err := gvr.RunLSTM(); err != nil {
		t.Fatal(err)
	}
}
