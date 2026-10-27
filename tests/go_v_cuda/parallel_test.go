package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaParallel — Go↔CUDA Parallel forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaParallel$'
func TestGoVCudaParallel(t *testing.T) {
	if err := gvc.RunParallel(); err != nil {
		t.Fatal(err)
	}
}
