package hf_test

import (
	"testing"

	"github.com/openfluke/welvet/model/hf"
)

func TestMaxSeqLenFromConfig(t *testing.T) {
	if got := hf.MaxSeqLenFromConfig(nil); got != 0 {
		t.Fatalf("nil → %d", got)
	}
	if got := hf.MaxSeqLenFromConfig(map[string]any{
		"max_position_embeddings": float64(32768),
	}); got != 32768 {
		t.Fatalf("max_position_embeddings → %d", got)
	}
	if got := hf.MaxSeqLenFromConfig(map[string]any{
		"n_positions": float64(1024),
	}); got != 1024 {
		t.Fatalf("n_positions → %d", got)
	}
}
