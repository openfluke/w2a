package go_v_qnn_test

import (
	"testing"

	gvq "github.com/openfluke/w2a/suites/go_v_qnn"
	"github.com/openfluke/welvet/nativeqnn"
)

// TestGoVQNN times Go CPU vs nativecpp vs Hexagon QNN on QNN-capable layers only.
func TestGoVQNN(t *testing.T) {
	if !nativeqnn.Available() {
		t.Skip("libwelvet_qnn not available (need cgo + windows/arm64 + DLL on PATH)")
	}
	if err := gvq.RunAll(); err != nil {
		t.Fatal(err)
	}
}

// TestGoVQNNMatrix runs README layer × dtype sweep and writes suites/go_v_qnn/RESULTS.md.
func TestGoVQNNMatrix(t *testing.T) {
	if !nativeqnn.Available() {
		t.Skip("libwelvet_qnn not available")
	}
	if err := gvq.RunOne(1); err != nil {
		t.Fatal(err)
	}
}
