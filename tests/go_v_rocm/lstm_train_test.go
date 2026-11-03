package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmLSTMTrain runs Go↔ROCm LSTM NormalBP (leaf + volumetric).
func TestGoVRocmLSTMTrain(t *testing.T) {
	if err := gvr.RunLSTMTrain(); err != nil {
		t.Fatal(err)
	}
}
