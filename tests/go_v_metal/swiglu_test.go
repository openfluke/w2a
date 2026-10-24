package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSwiGLU — Go↔Metal SwiGLU forward only (exact_bit + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalSwiGLU$'
func TestGoVMetalSwiGLU(t *testing.T) {
	if err := gvm.RunSwiGLU(); err != nil {
		t.Fatal(err)
	}
}
