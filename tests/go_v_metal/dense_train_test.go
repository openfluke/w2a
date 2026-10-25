package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalDenseTrain runs Go↔Metal Dense NormalBP (leaf + volumetric).
func TestGoVMetalDenseTrain(t *testing.T) {
	if err := gvm.RunDenseTrain(); err != nil {
		t.Fatal(err)
	}
}
