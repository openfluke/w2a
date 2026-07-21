package wav2vec2_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/model/wav2vec2"
	"github.com/openfluke/welvet/quant"
)

func cacheDir(t *testing.T) string {
	t.Helper()
	// welvet/.cache when running from w2a via replace ../
	candidates := []string{
		filepath.Join("..", "..", ".cache", "wav2vec2-base-960h"),
		filepath.Join("..", ".cache", "wav2vec2-base-960h"),
		filepath.Join(".cache", "wav2vec2-base-960h"),
	}
	// Also allow absolute from module root via env
	if d := os.Getenv("WELVET_WAV2VEC2_DIR"); d != "" {
		candidates = append([]string{d}, candidates...)
	}
	for _, d := range candidates {
		if st, err := os.Stat(filepath.Join(d, "model.safetensors")); err == nil && !st.IsDir() {
			return d
		}
	}
	t.Skip("wav2vec2-base-960h cache not found (set WELVET_WAV2VEC2_DIR)")
	return ""
}

func TestCTCVocabDecode(t *testing.T) {
	v := &wav2vec2.Vocab{
		IDToToken: []string{"<pad>", "<s>", "</s>", "<unk>", "|", "H", "I"},
		BlankID:   0,
	}
	got := v.DecodeCTCGreedy([]int{0, 5, 5, 6, 4, 5, 6, 0})
	if got != "HI HI" {
		t.Fatalf("got %q", got)
	}
}

func TestTranscribeSample3s(t *testing.T) {
	dir := cacheDir(t)
	wav := filepath.Join(dir, "sample3s.wav")
	if _, err := os.Stat(wav); err != nil {
		t.Skip("sample3s.wav missing")
	}
	m, err := wav2vec2.LoadHFDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	text, err := m.TranscribeFile(wav)
	if err != nil {
		t.Fatal(err)
	}
	want := "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAM"
	if text != want {
		t.Fatalf("transcript mismatch:\n got %q\nwant %q", text, want)
	}
	if !strings.HasPrefix(text, "GOING ALONG") {
		t.Fatalf("unexpected %q", text)
	}
}

func TestEntityRoundTrip(t *testing.T) {
	dir := cacheDir(t)
	ent := filepath.Join(dir, "roundtrip.entity")
	defer os.Remove(ent)
	if err := entity.PackFromHF(dir, ent, entity.PackOptions{
		Repo:   "facebook/wav2vec2-base-960h",
		Format: quant.FormatNone,
	}); err != nil {
		t.Fatal(err)
	}
	m, err := wav2vec2.LoadEntity(ent)
	if err != nil {
		t.Fatal(err)
	}
	wav := filepath.Join(dir, "sample3s.wav")
	if _, err := os.Stat(wav); err != nil {
		t.Skip("sample3s.wav missing")
	}
	text, err := m.TranscribeFile(wav)
	if err != nil {
		t.Fatal(err)
	}
	want := "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAM"
	if text != want {
		t.Fatalf("got %q want %q", text, want)
	}
}
