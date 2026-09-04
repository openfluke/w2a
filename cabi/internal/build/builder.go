// Polyglot C-ABI builder for Welvet (apps/w2a/cabi).
// Usage: go run builder.go -os linux -arch amd64
//        go run builder.go -os all
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type Platform struct {
	GOOS      string
	GOARCH    string
	GOARM     string
	DirName   string
	BuildMode string
}

func (p Platform) Ext() string {
	switch p.GOOS {
	case "windows":
		return ".dll"
	case "darwin":
		return ".dylib"
	case "ios":
		return ".a"
	default:
		return ".so"
	}
}

var allPlatforms = []Platform{
	{GOOS: "linux", GOARCH: "amd64", DirName: "linux_amd64", BuildMode: "c-shared"},
	{GOOS: "linux", GOARCH: "arm64", DirName: "linux_arm64", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "amd64", DirName: "macos_amd64", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "arm64", DirName: "macos_arm64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "amd64", DirName: "windows_amd64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "arm64", DirName: "windows_arm64", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "arm64", DirName: "android_arm64", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "amd64", DirName: "android_x86_64", BuildMode: "c-shared"},
	{GOOS: "ios", GOARCH: "arm64", DirName: "ios_arm64", BuildMode: "c-archive"},
}

func main() {
	targetOS := flag.String("os", runtime.GOOS, "windows|linux|darwin|android|ios|all")
	targetArch := flag.String("arch", runtime.GOARCH, "amd64|arm64|universal")
	outDir := flag.String("out", "dist", "Output directory")
	clean := flag.Bool("clean", false, "Remove output directory before building")
	test := flag.Bool("test", false, "Run cabi_verify after native build")
	soft := flag.Bool("soft", false, "Skip targets missing toolchain (do not fail)")
	flag.Parse()

	if *clean {
		fmt.Printf("Cleaning %s...\n", *outDir)
		_ = os.RemoveAll(*outDir)
	}

	var targets []Platform
	if *targetOS == "all" {
		targets = allPlatforms
	} else {
		targets = selectPlatforms(*targetOS, *targetArch)
	}

	var ok, skip, fail []string
	for _, p := range targets {
		err := buildPlatform(p, *outDir)
		if err != nil {
			if *soft || isToolchainSkip(err) {
				fmt.Printf("SKIP %s: %v\n", p.DirName, err)
				skip = append(skip, p.DirName)
				continue
			}
			fmt.Printf("FAILED %s: %v\n", p.DirName, err)
			fail = append(fail, p.DirName)
			continue
		}
		ok = append(ok, p.DirName)
		if *test && p.GOOS == runtime.GOOS && p.GOARCH == runtime.GOARCH {
			runVerify(p, *outDir)
		}
	}

	if *targetOS == "all" || (*targetOS == "darwin" && *targetArch == "universal") {
		if err := buildMacUniversal(*outDir); err != nil {
			if *soft {
				fmt.Printf("SKIP macos_universal: %v\n", err)
				skip = append(skip, "macos_universal")
			} else {
				fmt.Printf("FAILED macos_universal: %v\n", err)
				fail = append(fail, "macos_universal")
			}
		} else {
			ok = append(ok, "macos_universal")
		}
	}

	fmt.Printf("\n=== summary ok=%d skip=%d fail=%d ===\n", len(ok), len(skip), len(fail))
	if len(ok) > 0 {
		fmt.Println("OK:", strings.Join(ok, ", "))
	}
	if len(skip) > 0 {
		fmt.Println("SKIP:", strings.Join(skip, ", "))
	}
	if len(fail) > 0 {
		fmt.Println("FAIL:", strings.Join(fail, ", "))
		os.Exit(1)
	}
}

func selectPlatforms(goos, arch string) []Platform {
	switch arch {
	case "x86_64":
		arch = "amd64"
	case "aarch64":
		arch = "arm64"
	case "universal":
		// lipo step only — build both slices first when requested as "universal"
		if goos == "darwin" {
			var out []Platform
			for _, p := range allPlatforms {
				if p.GOOS == "darwin" {
					out = append(out, p)
				}
			}
			return out
		}
	}
	for _, p := range allPlatforms {
		if p.GOOS == goos && p.GOARCH == arch {
			return []Platform{p}
		}
	}
	return []Platform{{GOOS: goos, GOARCH: arch, DirName: goos + "_" + arch, BuildMode: "c-shared"}}
}

func isToolchainSkip(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "no cross-compiler") ||
		strings.Contains(msg, "NDK") ||
		strings.Contains(msg, "macOS only") ||
		strings.Contains(msg, "clang") && strings.Contains(msg, "not found")
}

func buildPlatform(p Platform, outBase string) error {
	fmt.Printf("\n--- Building %s ---\n", p.DirName)
	if p.GOOS == "ios" && runtime.GOOS != "darwin" {
		return fmt.Errorf("macOS only (iOS)")
	}
	if p.GOOS == "darwin" && runtime.GOOS != "darwin" && crossCC(p) == "" {
		return fmt.Errorf("no cross-compiler for darwin/%s", p.GOARCH)
	}

	outPath := filepath.Join(outBase, p.DirName)
	if err := os.MkdirAll(outPath, 0755); err != nil {
		return err
	}
	outFile := filepath.Join(outPath, "welvet"+p.Ext())
	cc := crossCC(p)
	if p.GOOS == "windows" && cc == "" {
		return fmt.Errorf("no cross-compiler for windows/%s (mingw / llvm-mingw)", p.GOARCH)
	}
	if p.GOOS == "android" && cc == "" {
		return fmt.Errorf("NDK not found (set ANDROID_NDK_HOME or install SDK under ~/Library/Android/sdk)")
	}
	if p.GOOS == "ios" && cc == "" {
		return fmt.Errorf("iOS SDK not found (install Xcode + iphoneos SDK)")
	}

	cmd := exec.Command("go", "build", "-buildmode="+p.BuildMode, "-o", outFile, "../../")
	env := append(cleanEnv(), "GOOS="+p.GOOS, "GOARCH="+p.GOARCH, "CGO_ENABLED=1")
	if p.GOARM != "" {
		env = append(env, "GOARM="+p.GOARM)
	}
	if cc != "" {
		env = append(env, "CC="+cc)
	}
	cflags, ldflags := cgoFlags(p)
	if cflags != "" {
		env = append(env, "CGO_CFLAGS="+cflags)
	}
	if ldflags != "" {
		env = append(env, "CGO_LDFLAGS="+ldflags)
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("%s\n", string(out))
		return fmt.Errorf("go build: %w", err)
	}
	fmt.Printf("  ✓  %s\n", outFile)
	h := strings.TrimSuffix(outFile, p.Ext()) + ".h"
	fmt.Printf("  ✓  %s\n", h)
	compileVerify(p, outPath, cc)
	return nil
}

func compileVerify(p Platform, outPath, cc string) {
	src := filepath.Join("..", "..", "test", "cabi_verify.c")
	if _, err := os.Stat(src); err != nil {
		return
	}
	exe := filepath.Join(outPath, "cabi_verify")
	if p.GOOS == "windows" {
		exe += ".exe"
	}
	compiler := cc
	if compiler == "" {
		compiler = "gcc"
		if _, err := exec.LookPath("gcc"); err != nil {
			compiler = "clang"
		}
	}
	parts := strings.Fields(compiler)
	bin := parts[0]
	args := append(parts[1:], "-I"+outPath, src, "-o", exe)
	switch p.GOOS {
	case "linux", "darwin":
		args = append(args, "-ldl")
	case "windows":
		args = append(args, filepath.Join(outPath, "welvet.dll"))
	default:
		return
	}
	fmt.Printf("  compiling cabi_verify...\n")
	out, err := exec.Command(bin, args...).CombinedOutput()
	if err != nil {
		fmt.Printf("  ⚠  cabi_verify: %s\n", strings.TrimSpace(string(out)))
		return
	}
	fmt.Printf("  ✓  %s\n", exe)
}

func runVerify(p Platform, outBase string) {
	exe := filepath.Join(outBase, p.DirName, "cabi_verify")
	lib := filepath.Join(outBase, p.DirName, "welvet"+p.Ext())
	cmd := exec.Command(exe, lib)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+filepath.Join(outBase, p.DirName))
	out, err := cmd.CombinedOutput()
	fmt.Print(string(out))
	if err != nil {
		fmt.Printf("cabi_verify failed: %v\n", err)
	}
}

func buildMacUniversal(outBase string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("macOS only")
	}
	a := filepath.Join(outBase, "macos_amd64", "welvet.dylib")
	b := filepath.Join(outBase, "macos_arm64", "welvet.dylib")
	if _, err := os.Stat(a); err != nil {
		return fmt.Errorf("need macos_amd64 first")
	}
	if _, err := os.Stat(b); err != nil {
		return fmt.Errorf("need macos_arm64 first")
	}
	outPath := filepath.Join(outBase, "macos_universal")
	_ = os.MkdirAll(outPath, 0755)
	out := filepath.Join(outPath, "welvet.dylib")
	cmd := exec.Command("lipo", "-create", a, b, "-output", out)
	if o, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("lipo: %s", string(o))
	}
	fmt.Printf("  ✓  %s\n", out)
	return nil
}

func cleanEnv() []string {
	out := make([]string, 0, len(os.Environ()))
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "CC=") ||
			strings.HasPrefix(e, "CGO_CFLAGS=") ||
			strings.HasPrefix(e, "CGO_LDFLAGS=") {
			continue
		}
		out = append(out, e)
	}
	return out
}

func look(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	return ""
}

func androidHome() string {
	for _, k := range []string{"ANDROID_HOME", "ANDROID_SDK_ROOT"} {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	if runtime.GOOS == "darwin" {
		home, _ := os.UserHomeDir()
		p := filepath.Join(home, "Library", "Android", "sdk")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	return ""
}

func ndkRoots() []string {
	roots := []string{
		os.Getenv("ANDROID_NDK_HOME"),
		os.Getenv("ANDROID_NDK_ROOT"),
	}
	if h := androidHome(); h != "" {
		ndk := filepath.Join(h, "ndk")
		entries, _ := os.ReadDir(ndk)
		// Prefer newest NDK (names are version-sorted ascending).
		for i := len(entries) - 1; i >= 0; i-- {
			if entries[i].IsDir() {
				roots = append(roots, filepath.Join(ndk, entries[i].Name()))
			}
		}
	}
	return roots
}

func ndkClang(triple string) string {
	for _, root := range ndkRoots() {
		if root == "" {
			continue
		}
		pre := filepath.Join(root, "toolchains", "llvm", "prebuilt")
		hosts, _ := os.ReadDir(pre)
		for _, h := range hosts {
			p := filepath.Join(pre, h.Name(), "bin", triple+"-clang")
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p
			}
		}
	}
	return ""
}

func iosSDKPath() string {
	out, err := exec.Command("xcrun", "--sdk", "iphoneos", "--show-sdk-path").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// cgoFlags returns platform-specific CGO_CFLAGS / CGO_LDFLAGS.
func cgoFlags(p Platform) (cflags, ldflags string) {
	switch p.GOOS {
	case "ios":
		sdk := iosSDKPath()
		if sdk == "" {
			return "", ""
		}
		arch := "arm64"
		if p.GOARCH == "amd64" {
			arch = "x86_64"
		}
		flags := fmt.Sprintf("-isysroot %s -arch %s -miphoneos-version-min=13.0", sdk, arch)
		return flags, flags
	case "windows":
		if p.GOARCH == "arm64" {
			return "", "-loleaut32 -lole32 -luuid"
		}
	}
	return "", ""
}

func crossCC(p Platform) string {
	host, hostArch := runtime.GOOS, runtime.GOARCH
	switch p.GOOS {
	case "linux":
		if host == "linux" && hostArch == p.GOARCH {
			return ""
		}
		if p.GOARCH == "amd64" {
			return look("x86_64-linux-gnu-gcc", "x86_64-unknown-linux-gnu-gcc")
		}
		if p.GOARCH == "arm64" {
			return look("aarch64-linux-gnu-gcc", "aarch64-unknown-linux-gnu-gcc")
		}
	case "darwin":
		if host == "darwin" {
			if hostArch == p.GOARCH {
				return ""
			}
			if p.GOARCH == "amd64" {
				return "clang -arch x86_64"
			}
			return "clang -arch arm64"
		}
	case "windows":
		if p.GOARCH == "amd64" {
			return look("x86_64-w64-mingw32-gcc")
		}
		if home := os.Getenv("LLVM_MINGW_HOME"); home != "" {
			p := filepath.Join(home, "bin", "aarch64-w64-mingw32-clang")
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
		return look("aarch64-w64-mingw32-clang", "aarch64-w64-mingw32-gcc")
	case "android":
		if p.GOARCH == "arm64" {
			return ndkClang("aarch64-linux-android21")
		}
		return ndkClang("x86_64-linux-android21")
	case "ios":
		if host == "darwin" {
			if iosSDKPath() == "" {
				return ""
			}
			return "clang"
		}
	}
	return ""
}
