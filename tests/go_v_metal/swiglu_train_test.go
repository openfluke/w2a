package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSwiGLUTrain runs Go↔Metal SwiGLU NormalBP (leaf + volumetric).
func TestGoVMetalSwiGLUTrain(t *testing.T) {
	if err := gvm.RunSwiGLUTrain(); err != nil {
		t.Fatal(err)
	}
}
