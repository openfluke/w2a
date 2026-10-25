package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSequentialTrain runs Go↔Metal Sequential NormalBP (leaf + volumetric).
func TestGoVMetalSequentialTrain(t *testing.T) {
	if err := gvm.RunSequentialTrain(); err != nil {
		t.Fatal(err)
	}
}
