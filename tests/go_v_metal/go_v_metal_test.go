package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalDenseForward — Go↔Metal Dense FormatNone×34 exact_bit + 3-way speed.
func TestGoVMetalDenseForward(t *testing.T) {
	if err := gvm.RunDenseForward(); err != nil {
		t.Fatal(err)
	}
}
