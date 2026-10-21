package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppEmbeddingTrain runs Go↔C++ Embedding NormalBP (leaf + volumetric).
func TestGoVCppEmbeddingTrain(t *testing.T) {
	if err := gvc.RunEmbeddingTrain(); err != nil {
		t.Fatal(err)
	}
}
