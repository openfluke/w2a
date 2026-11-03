package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmMamba — Go↔ROCm Mamba forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmMamba$'
func TestGoVRocmMamba(t *testing.T) {
	if err := gvr.RunMamba(); err != nil {
		t.Fatal(err)
	}
}
