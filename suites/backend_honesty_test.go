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
