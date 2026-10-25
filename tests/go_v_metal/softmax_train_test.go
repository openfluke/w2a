package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSoftmaxTrain runs Go↔Metal Softmax NormalBP (leaf + volumetric).
func TestGoVMetalSoftmaxTrain(t *testing.T) {
	if err := gvm.RunSoftmaxTrain(); err != nil {
		t.Fatal(err)
	}
}
