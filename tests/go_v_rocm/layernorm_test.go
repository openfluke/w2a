package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmLayerNorm — Go↔ROCm LayerNorm FormatNone forward (exact_bit + speed).
func TestGoVRocmLayerNorm(t *testing.T) {
	if err := gvr.RunLayerNorm(); err != nil {
		t.Fatal(err)
	}
}
