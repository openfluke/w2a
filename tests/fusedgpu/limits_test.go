package fusedgpu_test

import (
	"testing"

	"github.com/openfluke/welvet/fusedgpu"
)

func TestClampAttnMaxSeq(t *testing.T) {
	if got := fusedgpu.ClampAttnMaxSeq(0); got != fusedgpu.DefaultMaxSeq {
		t.Fatalf("unset → %d, want %d", got, fusedgpu.DefaultMaxSeq)
	}
	if got := fusedgpu.ClampAttnMaxSeq(512); got != 512 {
		t.Fatalf("512 → %d", got)
	}
	if got := fusedgpu.ClampAttnMaxSeq(100_000); got != fusedgpu.AttnScoresMaxSeq {
		t.Fatalf("huge → %d, want shader cap %d", got, fusedgpu.AttnScoresMaxSeq)
	}
}

func TestClampMaxSeqForKVBudget(t *testing.T) {
	// Tiny budget forces seq down below the shader cap.
	got := fusedgpu.ClampMaxSeqForKVBudget(2048, 14, 8, 128, 8<<20) // 8 MiB
	if got >= 2048 {
		t.Fatalf("expected budget clamp, got %d", got)
	}
	if got < 64 {
		t.Fatalf("seq collapsed too far: %d", got)
	}
	// No budget → shader clamp only.
	if got := fusedgpu.ClampMaxSeqForKVBudget(4096, 14, 8, 128, 0); got != fusedgpu.AttnScoresMaxSeq {
		t.Fatalf("no budget → %d, want %d", got, fusedgpu.AttnScoresMaxSeq)
	}
}

func TestEstimateHybridKVBytes(t *testing.T) {
	// 14 layers × 8 kv heads × 128 dim × 2048 seq × 8 bytes = 224 MiB
	got := fusedgpu.EstimateHybridKVBytes(14, 8, 128, 2048)
	want := int64(14 * 8 * 128 * 2048 * 8)
	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}
