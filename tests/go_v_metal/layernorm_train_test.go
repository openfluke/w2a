package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalLayerNormTrain runs Go↔Metal LayerNorm NormalBP (leaf + volumetric).
func TestGoVMetalLayerNormTrain(t *testing.T) {
	if err := gvm.RunLayerNormTrain(); err != nil {
		t.Fatal(err)
	}
}
