package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalCNNTrain runs Go↔Metal CNN1/2/3 NormalBP (leaf + volumetric).
func TestGoVMetalCNNTrain(t *testing.T) {
	if err := gvm.RunCNNTrain(); err != nil {
		t.Fatal(err)
	}
}
