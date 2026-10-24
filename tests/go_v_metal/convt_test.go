package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalConvT — Go↔Metal ConvT1/2/3 (exact_bit ×34 + timings).
func TestGoVMetalConvT(t *testing.T) {
	if err := gvm.RunConvT(); err != nil {
		t.Fatal(err)
	}
}
