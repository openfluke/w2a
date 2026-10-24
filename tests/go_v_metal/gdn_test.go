package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalGDN — Go↔Metal GDN Float32 (exact_bit + mid timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalGDN$'
func TestGoVMetalGDN(t *testing.T) {
	if err := gvm.RunGDN(); err != nil {
		t.Fatal(err)
	}
}
