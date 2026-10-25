package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalRMSNormTrain runs Go↔Metal RMSNorm NormalBP (leaf + volumetric).
func TestGoVMetalRMSNormTrain(t *testing.T) {
	if err := gvm.RunRMSNormTrain(); err != nil {
		t.Fatal(err)
	}
}
