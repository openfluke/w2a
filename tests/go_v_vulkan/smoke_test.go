package go_v_vulkan_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func denseFwdPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("WELVET_VULKAN_DENSE_FWD"); p != "" {
		return p
	}
	cand := filepath.Join("..", "..", "..", "welvet.vulkan", "build", "welvet_dense_fwd")
	if _, err := os.Stat(cand); err != nil {
		t.Skip("welvet.vulkan not built (set WELVET_VULKAN_DENSE_FWD or run welvet.vulkan/scripts/build_linux.sh)")
	}
	return cand
}

func TestGoVVulkanSmokeVersion(t *testing.T) {
	bin := denseFwdPath(t)
	out, err := exec.Command(bin, "--version").CombinedOutput()
	if err != nil {
		t.Fatalf("version: %v\n%s", err, out)
	}
	if len(out) == 0 {
		t.Fatal("empty version")
	}
	t.Logf("version: %s", string(out))
}

func TestGoVVulkanSmokeGaps(t *testing.T) {
	bin := denseFwdPath(t)
	out, err := exec.Command(bin, "--gaps").CombinedOutput()
	if err != nil {
		t.Fatalf("gaps: %v\n%s", err, out)
	}
	t.Logf("%s", string(out))
}
