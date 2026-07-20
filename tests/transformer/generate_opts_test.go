package transformer_test

import (
	"testing"

	"github.com/openfluke/welvet/model/sampling"
	"github.com/openfluke/welvet/model/transformer"
)

func TestGenOptionsSampleTopKDefaults(t *testing.T) {
	// Documented defaults: Temperature 0 / TopK 0 → greedy via SampleTopK.
	logits := []float32{0.1, 4.0, 0.2}
	opts := transformer.GenOptions{}
	next := sampling.SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)
	if next != 1 {
		t.Fatalf("default sample: got %d want 1", next)
	}
	opts.Temperature = 0.9
	opts.TopK = 1
	next = sampling.SampleTopK(logits, opts.TopK, opts.Temperature, opts.Deterministic)
	if next != 1 {
		t.Fatalf("TopK=1: got %d want 1", next)
	}
}
