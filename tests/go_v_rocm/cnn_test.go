package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmCNN — Go↔ROCm CNN1/2/3 FormatNone×34 exact_bit + timings.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmCNN$'
func TestGoVRocmCNN(t *testing.T) {
	if err := gvr.RunCNN(); err != nil {
		t.Fatal(err)
	}
}
