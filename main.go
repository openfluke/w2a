package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/openfluke/w2a/suites"
	denssuite "github.com/openfluke/w2a/suites/dense"
	mhasuite "github.com/openfluke/w2a/suites/mha"
	lnsuite "github.com/openfluke/w2a/suites/layernorm"
	cnn1suite "github.com/openfluke/w2a/suites/cnn1"
	cnn2suite "github.com/openfluke/w2a/suites/cnn2"
	cnn3suite "github.com/openfluke/w2a/suites/cnn3"
	rnnsuite "github.com/openfluke/w2a/suites/rnn"
	lstmsuite "github.com/openfluke/w2a/suites/lstm"
	embeddingsuite "github.com/openfluke/w2a/suites/embedding"
	softmaxsuite "github.com/openfluke/w2a/suites/softmax"
	rmsnsuite "github.com/openfluke/w2a/suites/rmsnorm"
	swigsuite "github.com/openfluke/w2a/suites/swiglu"
)

type suite struct {
	Name string
	Desc string
	Run  func() error
	Menu func() // optional sub-menu (e.g. pick individual cases)
}

func main() {
	allSuites := []suite{
		{
			Name: "Dense",
			Desc: "34 dtypes, timed matrix, gap census (see welvet README)",
			Run:  denssuite.RunAll,
			Menu: denseSubmenu,
		},
		{
			Name: "MHA",
			Desc: "causal+RoPE+GQA; FormatNone×34 + quants × backends + train grids",
			Run:  mhasuite.RunAll,
			Menu: mhaSubmenu,
		},
		{
			Name: "SwiGLU",
			Desc: "SiLU-gated FFN; FormatNone×34 + quants × backends + train grids",
			Run:  swigsuite.RunAll,
			Menu: swigluSubmenu,
		},
		{
			Name: "RMSNorm",
			Desc: "γ-scale RMSNorm; FormatNone×34 + quants × backends + train grids",
			Run:  rmsnsuite.RunAll,
			Menu: rmsnormSubmenu,
		},
		{
			Name: "LayerNorm",
			Desc: "γ+β LayerNorm; FormatNone×34 + quants × backends + train grids",
			Run:  lnsuite.RunAll,
			Menu: layernormSubmenu,
		},
		{
			Name: "CNN1",
			Desc: "Conv1d im2col→Dense; FormatNone×34 + quants × backends + train grids",
			Run:  cnn1suite.RunAll,
			Menu: cnn1Submenu,
		},
		{
			Name: "CNN2",
			Desc: "Conv2d im2col→Dense; FormatNone×34 + quants × backends + train grids",
			Run:  cnn2suite.RunAll,
			Menu: cnn2Submenu,
		},
		{
			Name: "CNN3",
			Desc: "Conv3d im2col→Dense; FormatNone×34 + quants × backends + train grids",
			Run:  cnn3suite.RunAll,
			Menu: cnn3Submenu,
		},
		{
			Name: "RNN",
			Desc: "vanilla tanh RNN; FormatNone×34 + quants × backends + train grids",
			Run:  rnnsuite.RunAll,
			Menu: rnnSubmenu,
		},
		{
			Name: "LSTM",
			Desc: "LSTM i/f/g/o; FormatNone×34 + quants × backends + train grids",
			Run:  lstmsuite.RunAll,
			Menu: lstmSubmenu,
		},
		{
			Name: "Embedding",
			Desc: "token gather/scatter; FormatNone×34 + quants × backends + train grids",
			Run:  embeddingsuite.RunAll,
			Menu: embeddingSubmenu,
		},
		{
			Name: "Softmax",
			Desc: "weightless last-axis Softmax; ALU × backends + train grids",
			Run:  softmaxsuite.RunAll,
			Menu: softmaxSubmenu,
		},
		// Add more suites here as layers land (residual, …).
	}

	in := bufio.NewReader(os.Stdin)
	for {
		fmt.Println()
		fmt.Println("w2a — Welvet test harness")
		fmt.Println("  [0] Run ALL suites")
		for i, s := range allSuites {
			fmt.Printf("  [%d] %s — %s\n", i+1, s.Name, s.Desc)
		}
		fmt.Println("  [q] Quit")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "q" || line == "Q" || line == "quit" {
			return
		}
		if line == "0" {
			_ = withSuiteLog(func() error {
				var failed int
				for _, s := range allSuites {
					fmt.Printf("\n▶ %s\n", s.Name)
					if err := s.Run(); err != nil {
						fmt.Printf("❌ %s: %v\n", s.Name, err)
						failed++
					} else {
						fmt.Printf("✅ %s: all PASS\n", s.Name)
					}
				}
				if failed > 0 {
					fmt.Printf("\n%d suite(s) failed\n", failed)
					return fmt.Errorf("%d suite(s) failed", failed)
				}
				fmt.Println("\nAll suites PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(allSuites) {
			fmt.Println("Invalid choice")
			continue
		}
		s := allSuites[n-1]
		if s.Menu != nil {
			s.Menu()
			continue
		}
		fmt.Printf("\n▶ %s\n", s.Name)
		_ = withSuiteLog(func() error {
			if err := s.Run(); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			fmt.Printf("✅ %s: all PASS\n", s.Name)
			return nil
		})
	}
}

func denseSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := denssuite.Cases()
	for {
		fmt.Println()
		fmt.Println("Dense suite")
		fmt.Println("  [0] Run ALL dense cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := denssuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ Dense: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := denssuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func mhaSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := mhasuite.Cases()
	for {
		fmt.Println()
		fmt.Println("MHA suite")
		fmt.Println("  [0] Run ALL MHA cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := mhasuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ MHA: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := mhasuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func swigluSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := swigsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("SwiGLU suite")
		fmt.Println("  [0] Run ALL SwiGLU cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := swigsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ SwiGLU: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := swigsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func rmsnormSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := rmsnsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("RMSNorm suite")
		fmt.Println("  [0] Run ALL RMSNorm cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := rmsnsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ RMSNorm: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := rmsnsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func layernormSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := lnsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("LayerNorm suite")
		fmt.Println("  [0] Run ALL LayerNorm cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := lnsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ LayerNorm: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := lnsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func cnn1Submenu() {
	in := bufio.NewReader(os.Stdin)
	cases := cnn1suite.Cases()
	for {
		fmt.Println()
		fmt.Println("CNN1 suite")
		fmt.Println("  [0] Run ALL CNN1 cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := cnn1suite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ CNN1: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := cnn1suite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func cnn2Submenu() {
	in := bufio.NewReader(os.Stdin)
	cases := cnn2suite.Cases()
	for {
		fmt.Println()
		fmt.Println("CNN2 suite")
		fmt.Println("  [0] Run ALL CNN2 cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := cnn2suite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ CNN2: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := cnn2suite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func cnn3Submenu() {
	in := bufio.NewReader(os.Stdin)
	cases := cnn3suite.Cases()
	for {
		fmt.Println()
		fmt.Println("CNN3 suite")
		fmt.Println("  [0] Run ALL CNN3 cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := cnn3suite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ CNN3: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := cnn3suite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func rnnSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := rnnsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("RNN suite")
		fmt.Println("  [0] Run ALL RNN cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := rnnsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ RNN: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := rnnsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func lstmSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := lstmsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("LSTM suite")
		fmt.Println("  [0] Run ALL LSTM cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := lstmsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ LSTM: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := lstmsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func embeddingSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := embeddingsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("Embedding suite")
		fmt.Println("  [0] Run ALL Embedding cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := embeddingsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ Embedding: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := embeddingsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

func softmaxSubmenu() {
	in := bufio.NewReader(os.Stdin)
	cases := softmaxsuite.Cases()
	for {
		fmt.Println()
		fmt.Println("Softmax suite")
		fmt.Println("  [0] Run ALL Softmax cases")
		for i, c := range cases {
			fmt.Printf("  [%d] %s\n", i+1, c.Name)
		}
		fmt.Println("  [b] Back")
		fmt.Print("Choice: ")

		line, err := readLine(in)
		if err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "b" || line == "B" || line == "back" {
			return
		}
		if line == "0" {
			fmt.Println()
			_ = withSuiteLog(func() error {
				if err := softmaxsuite.RunAll(); err != nil {
					fmt.Printf("❌ %v\n", err)
					return err
				}
				fmt.Println("✅ Softmax: all PASS")
				return nil
			})
			continue
		}
		n, err := strconv.Atoi(line)
		if err != nil || n < 1 || n > len(cases) {
			fmt.Println("Invalid choice")
			continue
		}
		fmt.Println()
		_ = withSuiteLog(func() error {
			if err := softmaxsuite.RunOne(n); err != nil {
				fmt.Printf("❌ %v\n", err)
				return err
			}
			return nil
		})
	}
}

// withSuiteLog clears w2a/logs/ to a single suite.txt, tees stdout, prints the
// grand cell count + by-layer / by-op tables when done. Same path for Dense
// alone or "run ALL layers" later.
func withSuiteLog(fn func() error) error {
	restore, err := suites.BeginLog()
	if err != nil {
		fmt.Printf("suite log: %v\n", err)
		return err
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()
	return fn()
}

func readLine(in *bufio.Reader) (string, error) {
	line, err := in.ReadString('\n')
	if err != nil && len(strings.TrimSpace(line)) == 0 {
		return "", err
	}
	return strings.TrimRight(line, "\r\n"), nil
}
