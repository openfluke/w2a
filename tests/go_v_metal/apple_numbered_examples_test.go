package go_v_metal_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAppleNumberedExamples(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("Apple GPU examples require macOS/arm64")
	}
	root := os.Getenv("WELVET_APPLE_EXAMPLES_SOURCE")
	if root == "" {
		root = "../../../example/welvet"
	}
	root, e := filepath.Abs(root)
	if e != nil {
		t.Fatal(e)
	}
	cases := []struct {
		slug, expected string
		args           []string
	}{
		{"82-apple-metal", "output=[0.5 -0.25]", nil},
		{"83-apple-mamba-dtypes", "Mamba dtypes=34", nil},
		{"84-apple-training", "mode=StepTweenSplitSparse", []string{"-mode", "all"}},
		{"85-apple-save-weights", "reload=exact", []string{"-out", filepath.Join(t.TempDir(), "trained.entity")}},
		{"86-apple-json-routing", "metal routing={Mounts:2", nil},
		{"87-apple-mixed-backends", "cpp calls=", nil},
		{"88-apple-mamba-results", "not a live benchmark", nil},
	}
	for _, c := range cases {
		t.Run(c.slug, func(t *testing.T) {
			args := append([]string{"run", "."}, c.args...)
			cmd := exec.Command("go", args...)
			cmd.Dir = filepath.Join(root, c.slug)
			out, e := cmd.CombinedOutput()
			if e != nil {
				t.Fatalf("%v\n%s", e, out)
			}
			if strings.Contains(string(out), "SKIP:") {
				t.Fatalf("required native path skipped: %s", out)
			}
			if !strings.Contains(string(out), c.expected) {
				t.Fatalf("missing %q: %s", c.expected, out)
			}
			if c.slug == "84-apple-training" && strings.Count(string(out), "mode=") != 29 {
				t.Fatalf("expected 29 training modes: %s", out)
			}
		})
	}
	t.Run("chapter_index", func(t *testing.T) {
		raw, e := os.ReadFile(filepath.Join(root, "_chapters.json"))
		if e != nil {
			t.Fatal(e)
		}
		var chapters []struct {
			Slug    string
			Num     string
			HasMain bool `json:"has_main"`
		}
		if e = json.Unmarshal(raw, &chapters); e != nil {
			t.Fatal(e)
		}
		for _, c := range cases {
			count := 0
			for _, ch := range chapters {
				if ch.Slug == c.slug && ch.HasMain {
					count++
				}
			}
			if count != 1 {
				t.Fatalf("chapter %s appears %d times", c.slug, count)
			}
		}
	})
}
