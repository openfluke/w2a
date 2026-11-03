package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmGDN — Go↔ROCm GDN forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmGDN$'
func TestGoVRocmGDN(t *testing.T) {
	if err := gvr.RunGDN(); err != nil {
		t.Fatal(err)
	}
}
