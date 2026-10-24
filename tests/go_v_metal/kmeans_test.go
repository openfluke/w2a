package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalKMeans — Go↔Metal KMeans soft assign (exact_bit ×34 both modes + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalKMeans$'
func TestGoVMetalKMeans(t *testing.T) {
	if err := gvm.RunKMeans(); err != nil {
		t.Fatal(err)
	}
}
