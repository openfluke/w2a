package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalConvTTrain runs Go↔Metal ConvT1/2/3 NormalBP (leaf + volumetric).
func TestGoVMetalConvTTrain(t *testing.T) {
	if err := gvm.RunConvTTrain(); err != nil {
		t.Fatal(err)
	}
}
