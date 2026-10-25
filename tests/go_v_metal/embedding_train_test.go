package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalEmbeddingTrain runs Go↔Metal Embedding NormalBP (leaf + volumetric).
func TestGoVMetalEmbeddingTrain(t *testing.T) {
	if err := gvm.RunEmbeddingTrain(); err != nil {
		t.Fatal(err)
	}
}
