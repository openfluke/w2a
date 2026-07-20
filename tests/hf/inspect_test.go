package hf_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/openfluke/welvet/model/hf"
)

func TestDetectArchitectureLlamaStyle(t *testing.T) {
	cfg := map[string]any{
		"model_type":            "llama",
		"num_hidden_layers":     float64(2),
		"hidden_size":           float64(64),
		"num_attention_heads":   float64(4),
		"num_key_value_heads":   float64(2),
		"intermediate_size":     float64(128),
		"vocab_size":            float64(100),
		"eos_token_id":          float64(2),
	}
	if kind := hf.DetectArchitecture(cfg); kind != hf.ArchLlamaStyleDecoder {
		t.Fatalf("kind=%v", kind)
	}
	dims, err := hf.ParseDecoderDims(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if dims.NumLayers != 2 || dims.HiddenSize != 64 || dims.VocabSize != 100 {
		t.Fatalf("dims=%+v", dims)
	}
	eos := hf.EOSTokenIDs(cfg)
	if len(eos) != 1 || eos[0] != 2 {
		t.Fatalf("eos=%v", eos)
	}
}

func TestInspectSnapshot(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
  "model_type": "llama",
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "intermediate_size": 128,
  "vocab_size": 100,
  "eos_token_id": 2
}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	info, err := hf.InspectSnapshot(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Architecture != hf.ArchLlamaStyleDecoder {
		t.Fatalf("arch=%v", info.Architecture)
	}
	if info.Dims.HiddenSize != 64 || info.Dims.NumLayers != 2 {
		t.Fatalf("dims=%+v", info.Dims)
	}
	if len(info.EOSTokens) != 1 || info.EOSTokens[0] != 2 {
		t.Fatalf("eos=%v", info.EOSTokens)
	}
}
