package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSoftmaxTrain runs Go↔ROCm Softmax NormalBP (leaf + volumetric).
func TestGoVRocmSoftmaxTrain(t *testing.T) {
	if err := gvr.RunSoftmaxTrain(); err != nil {
		t.Fatal(err)
	}
}
