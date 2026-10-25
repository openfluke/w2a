package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalLSTMTrain runs Go↔Metal LSTM NormalBP (leaf + volumetric).
func TestGoVMetalLSTMTrain(t *testing.T) {
	if err := gvm.RunLSTMTrain(); err != nil {
		t.Fatal(err)
	}
}
