package go_v_mlx_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_mlx"
	"github.com/openfluke/welvet/nativemlx"
)

// TestGoVMLX times Go CPU vs Apple MLX on MLX-capable forward layers.
func TestGoVMLX(t *testing.T) {
	if !nativemlx.Available() {
		t.Skip("libwelvet_mlx not available (need cgo + darwin/arm64 + dylib + Metal)")
	}
	if err := gvm.RunAll(); err != nil {
		t.Fatal(err)
	}
}

// TestGoVMLXCaps runs the caps inventory case only.
func TestGoVMLXCaps(t *testing.T) {
	if !nativemlx.Available() {
		t.Skip("libwelvet_mlx not available")
	}
	if err := gvm.RunOne(1); err != nil {
		t.Fatal(err)
	}
}
