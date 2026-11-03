package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmRMSNorm — Go↔ROCm RMSNorm FormatNone forward (exact_bit + speed).
func TestGoVRocmRMSNorm(t *testing.T) {
	if err := gvr.RunRMSNorm(); err != nil {
		t.Fatal(err)
	}
}
