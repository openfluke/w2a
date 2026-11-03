package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmDenseTrain runs Go↔ROCm Dense NormalBP (leaf + volumetric).
func TestGoVRocmDenseTrain(t *testing.T) {
	if err := gvr.RunDenseTrain(); err != nil {
		t.Fatal(err)
	}
}
