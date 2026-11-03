package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmMHATrain runs Go↔ROCm MHA NormalBP (leaf + volumetric).
func TestGoVRocmMHATrain(t *testing.T) {
	if err := gvr.RunMHATrain(); err != nil {
		t.Fatal(err)
	}
}
