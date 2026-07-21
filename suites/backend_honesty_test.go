package suites

import "testing"

func TestAffinePackable(t *testing.T) {
	if AffinePackable(8, 16) {
		t.Fatal("16 cols should not pack")
	}
	if !AffinePackable(8, 64) {
		t.Fatal("64 cols should pack")
	}
}

func TestStampWebGPUNote(t *testing.T) {
	st, nt := StampWebGPUNote("rmsnorm", true, "OK", "")
	if st != "OK" || nt == "" {
		t.Fatalf("got %s %q", st, nt)
	}
	st, nt = StampWebGPUNote("dense", true, "OK", "keep-me")
	if nt != "keep-me" {
		t.Fatalf("should keep existing note, got %q", nt)
	}
}

func TestStampBackendNoteSIMD(t *testing.T) {
	st, nt := StampBackendNote("dense", true, false, "OK", "")
	if st != "OK" || nt == "" {
		t.Fatalf("got %s %q", st, nt)
	}
}

func TestWebGPUKindPeakLayers(t *testing.T) {
	for _, layer := range []string{
		"mha", "swiglu", "softmax", "layernorm", "rmsnorm",
		"cnn1", "embedding", "rnn", "lstm", "dense",
	} {
		st, nt := WebGPUKind(layer)
		if st != "OK" || nt == "" {
			t.Fatalf("%s: got %s %q", layer, st, nt)
		}
	}
}
