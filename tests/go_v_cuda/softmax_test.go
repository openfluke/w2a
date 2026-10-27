package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSoftmax — Go↔CUDA Softmax forward (all Kind ×34 exact_bit + timings).
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaSoftmax$'
func TestGoVCudaSoftmax(t *testing.T) {
	if err := gvc.RunSoftmax(); err != nil {
		t.Fatal(err)
	}
}
