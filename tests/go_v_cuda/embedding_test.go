package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaEmbedding — Go↔CUDA Embedding gather (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaEmbedding$'
func TestGoVCudaEmbedding(t *testing.T) {
	if err := gvc.RunEmbedding(); err != nil {
		t.Fatal(err)
	}
}
