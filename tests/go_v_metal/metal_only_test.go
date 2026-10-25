package go_v_metal_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// This gate checks GPU primitives, native updates, activation derivatives, cache
// coherence and rejected host math. NormalBP layer coverage has its own grouped gate.
func TestGoVMetalOnly(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Apple Metal requires macOS")
	}
	_, file, _, _ := runtime.Caller(0)
	metal := filepath.Clean(filepath.Join(filepath.Dir(file), "../../../welvet.metal"))
	build := filepath.Join(metal, "build")
	if path := os.Getenv("WELVET_METAL_DENSE_FWD"); path != "" {
		build = filepath.Dir(path)
	}
	probe := filepath.Join(t.TempDir(), "metal_only")
	cmd := exec.Command("clang++", "-std=c++20", "-I"+filepath.Join(metal, "include"),
		filepath.Join(filepath.Dir(file), "testdata/metal_only.cpp"), "-L"+build,
		"-lwelvet_metal", "-Wl,-rpath,"+build, "-o", probe)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile probe: %v\n%s", err, output)
	}
	// Run from an unrelated directory to verify embedded shader delivery.
	cmd = exec.Command(probe)
	cmd.Dir = t.TempDir()
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Metal-only probe: %v\n%s", err, output)
	} else {
		t.Log(string(output))
	}
	for _, c := range gvm.CasesDenseTrain() {
		if strings.Contains(c.Name, "all 34") {
			continue
		}
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, c := range gvm.CasesStepBPTrain() {
		if strings.Contains(c.Name, "Float32") && (strings.HasPrefix(c.Name, "Dense ") || strings.HasPrefix(c.Name, "Sequential ") || strings.HasPrefix(c.Name, "NestedStack ")) {
			t.Run(c.Name, func(t *testing.T) {
				if err := c.Run(); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}
