package go_v_metal_test

import (
	"bufio"
	"encoding/json"
	"github.com/openfluke/welvet/nativecpp"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestAppleMambaStress(t *testing.T) {
	source := os.Getenv("WELVET_APPLE_STRESS_SOURCE")
	if source == "" {
		source = "../../../example/welvet/apple_stress_test"
	}
	source, e := filepath.Abs(source)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(filepath.Join(source, "main.go")); e != nil {
		t.Fatal("Apple Mamba example source unavailable:", e)
	}
	if !nativecpp.NativeCPUAvailable() {
		t.Skip("requires normal cgo build with C++ artifact")
	}
	binary := filepath.Join(t.TempDir(), "apple_stress_test")
	build := exec.Command("go", "build", "-o", binary, ".")
	build.Dir = source
	if out, e := build.CombinedOutput(); e != nil {
		t.Fatalf("build: %v\n%s", e, out)
	}
	t.Run("full_plan", func(t *testing.T) {
		out, e := exec.Command(binary, "-list").CombinedOutput()
		if e != nil {
			t.Fatalf("list: %v %s", e, out)
		}
		var plan struct {
			Layers, DTypes, Modes []string
			Selected              int `json:"selected_rows"`
		}
		if e = json.Unmarshal(out, &plan); e != nil {
			t.Fatal(e)
		}
		if len(plan.Layers) != 1 || plan.Layers[0] != "mamba" || len(plan.DTypes) != 34 || len(plan.Modes) != 29 || plan.Selected != 1020 {
			t.Fatalf("incomplete plan: %+v", plan)
		}
	})
	t.Run("invalid_selection", func(t *testing.T) {
		if e := exec.Command(binary, "-list", "-modes", "not-a-mode").Run(); e == nil {
			t.Fatal("unknown mode silently accepted")
		}
		if e := exec.Command(binary, "-list", "-lr", "NaN").Run(); e == nil {
			t.Fatal("non-finite learning rate accepted")
		}
	})
	t.Run("all_modes_real_bindings", func(t *testing.T) {
		dir := t.TempDir()
		cmd := exec.Command(binary, "-profile", "smoke", "-dtypes", "float32", "-samples", "1", "-iters", "1", "-steps", "8", "-warmup", "1", "-out", dir)
		if out, e := cmd.CombinedOutput(); e != nil {
			t.Fatalf("smoke: %v\n%s", e, out)
		}
		file, e := os.Open(filepath.Join(dir, "results.jsonl"))
		if e != nil {
			t.Fatal(e)
		}
		defer file.Close()
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 4096), 1<<20)
		rows := 0
		modes := map[string]bool{}
		for scanner.Scan() {
			var envelope struct {
				Type string
				Data json.RawMessage
			}
			if e = json.Unmarshal(scanner.Bytes(), &envelope); e != nil {
				t.Fatal(e)
			}
			if envelope.Type != "result" {
				continue
			}
			var row struct {
				Phase, Layer, Mode, Status string
				Results                    []struct {
					Backend, Error string
					Calls          uint64 `json:"successful_cpp_calls"`
				}
				Pairs []json.RawMessage
			}
			if e = json.Unmarshal(envelope.Data, &row); e != nil {
				t.Fatal(e)
			}
			if row.Layer != "mamba" || row.Status == "error" || len(row.Results) != 3 || len(row.Pairs) != 3 {
				t.Fatalf("invalid result: %s", envelope.Data)
			}
			for _, r := range row.Results {
				if r.Error != "" {
					t.Fatal(r.Error)
				}
				if r.Backend == "cpp" && r.Calls == 0 {
					t.Fatal("C++ row did not call native code")
				}
			}
			if row.Phase == "train" {
				modes[row.Mode] = true
			}
			rows++
		}
		if e = scanner.Err(); e != nil {
			t.Fatal(e)
		}
		if rows != 30 || len(modes) != 29 {
			t.Fatalf("rows=%d modes=%d", rows, len(modes))
		}
	})
}
