package sampling_test

import (
	"strings"
	"testing"

	"github.com/openfluke/welvet/model/sampling"
)

func TestArgMaxAndSampleTopKGreedy(t *testing.T) {
	logits := []float32{0.1, 3.0, 0.2, 1.5}
	if got := sampling.ArgMax(logits); got != 1 {
		t.Fatalf("ArgMax: got %d want 1", got)
	}
	if got := sampling.SampleTopK(logits, 1, 1.0, false); got != 1 {
		t.Fatalf("SampleTopK topK=1: got %d want 1", got)
	}
	if got := sampling.SampleTopK(logits, 4, 0, false); got != 1 {
		t.Fatalf("SampleTopK temp=0: got %d want 1", got)
	}
	if got := sampling.SampleTopK(logits, 4, 0.8, true); got != 1 {
		t.Fatalf("SampleTopK deterministic: got %d want 1", got)
	}
}

func TestSampleTopKRestricted(t *testing.T) {
	// Make index 3 clearly best among top-2 after temp scale; index 0/2 are low.
	logits := []float32{-10, 0.5, -10, 2.0}
	got := sampling.SampleTopK(logits, 2, 0.01, false)
	if got != 1 && got != 3 {
		t.Fatalf("SampleTopK topK=2: got %d want 1 or 3", got)
	}
}

func TestBanAndPenalty(t *testing.T) {
	logits := []float32{1, 5, 2}
	sampling.BanIDs(logits, []int{1})
	if sampling.ArgMax(logits) == 1 {
		t.Fatal("BanIDs failed")
	}
	logits = []float32{2, 2, 2}
	sampling.ApplyRepetitionPenalty(logits, []uint32{0}, 1.5, 8)
	if !(logits[0] < logits[1]) {
		t.Fatalf("penalty: got %v", logits)
	}
}

func TestReplyLooksDegenerate(t *testing.T) {
	ok := "I'm fine. Just trying to help out! How can I assist you today?"
	if sampling.ReplyLooksDegenerate(ok) {
		t.Fatalf("clean reply flagged: %q", ok)
	}
	junk := "everwastewaygroundwill happen thisï¼againstyardWISEBAOULDWAULEAVEGBATHugeryawaybackwardArgumentable"
	if !sampling.ReplyLooksDegenerate(junk) {
		t.Fatalf("junk reply not flagged: %q", junk)
	}
	san := sampling.SanitizeChatReply("Hello there. This is fine. " + junk)
	if san == "" || sampling.ReplyLooksDegenerate(san) {
		t.Fatalf("sanitize failed: %q", san)
	}
	if !strings.Contains(san, "This is fine.") {
		t.Fatalf("sanitize dropped clean sentence: %q", san)
	}
}
