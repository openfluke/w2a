//go:build welvet_cuda

package go_v_cuda_test

import (
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativecuda"
	"strings"
	"testing"
)

func TestGoNativeCudaIndependentLink(t *testing.T) {
	if nativecpp.NativeCPUAvailable() {
		t.Fatal("CUDA-only build enabled CPU sibling")
	}
	_, model, x, _ := cudaLifecycleModel(t)
	s, e := nativecuda.Mount(model)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	if !strings.HasPrefix(s.Identity(), "cuda:") {
		t.Fatalf("wrong backend: %s", s.Identity())
	}
	if _, e = s.Forward(x); e != nil {
		t.Fatal(e)
	}
}
