package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmEmbeddingTrain runs Go↔ROCm Embedding NormalBP (leaf + volumetric).
func TestGoVRocmEmbeddingTrain(t *testing.T) {
	if err := gvr.RunEmbeddingTrain(); err != nil {
		t.Fatal(err)
	}
}
