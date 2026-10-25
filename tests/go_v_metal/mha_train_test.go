package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalMHATrain runs Go↔Metal MHA NormalBP (leaf + volumetric).
func TestGoVMetalMHATrain(t *testing.T) {
	if err := gvm.RunMHATrain(); err != nil {
		t.Fatal(err)
	}
}
