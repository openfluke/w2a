package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaKMeans — Go↔CUDA KMeans forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaKMeans$'
func TestGoVCudaKMeans(t *testing.T) {
	if err := gvc.RunKMeans(); err != nil {
		t.Fatal(err)
	}
}
