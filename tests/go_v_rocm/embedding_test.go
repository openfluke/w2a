package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmEmbedding — Go↔ROCm Embedding gather (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmEmbedding$'
func TestGoVRocmEmbedding(t *testing.T) {
	if err := gvr.RunEmbedding(); err != nil {
		t.Fatal(err)
	}
}
