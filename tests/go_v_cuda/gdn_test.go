package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaGDN — Go↔CUDA GDN forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaGDN$'
func TestGoVCudaGDN(t *testing.T) {
	if err := gvc.RunGDN(); err != nil {
		t.Fatal(err)
	}
}
