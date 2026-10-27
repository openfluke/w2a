package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaMamba — Go↔CUDA Mamba forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaMamba$'
func TestGoVCudaMamba(t *testing.T) {
	if err := gvc.RunMamba(); err != nil {
		t.Fatal(err)
	}
}
