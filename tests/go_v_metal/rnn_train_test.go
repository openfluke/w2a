package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalRNNTrain runs Go↔Metal RNN NormalBP (leaf + volumetric).
func TestGoVMetalRNNTrain(t *testing.T) {
	if err := gvm.RunRNNTrain(); err != nil {
		t.Fatal(err)
	}
}
