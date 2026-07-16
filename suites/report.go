// Package suites holds cross-layer harness helpers (logging, cell counts, end report).
package suites

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Cell is one atomic matrix check (dtype×format×backend×grid×op).
type Cell struct {
	Layer   string // dense, mha, …
	Op      string // fwd, bwd, train, census, …
	DType   string
	Format  string
	Backend string
	Grid    string // e.g. 1x1x1, "-" 
	Status  string // OK, GAP, FAIL
	Note    string
}

// CaseRow is one named suite case (go subtest).
type CaseRow struct {
	Layer  string
	Name   string
	Status string // PASS, FAIL
	Err    string
	Cells  int // cells recorded during this case
}

const flushToken = "\x1eW2A_LOG_FLUSH\x1e\n"

var (
	mu       sync.Mutex
	cells    []Cell
	cases    []CaseRow
	started  time.Time
	logFile  *os.File
	logOld   *os.File
	logPipeW *os.File
	logDone  chan struct{}
	flushCh  chan struct{}
	caseMark int // cells len at case start
)

// Reset clears counters (call at suite start before logging).
func Reset() {
	mu.Lock()
	defer mu.Unlock()
	cells = cells[:0]
	cases = cases[:0]
	started = time.Now()
	caseMark = 0
}

// RecordCell adds one matrix cell result.
func RecordCell(c Cell) {
	mu.Lock()
	defer mu.Unlock()
	if c.Grid == "" {
		c.Grid = "-"
	}
	if c.DType == "" {
		c.DType = "-"
	}
	if c.Format == "" {
		c.Format = "-"
	}
	cells = append(cells, c)
}

// BeginCase marks the start of a named suite case for cell attribution.
func BeginCase() {
	mu.Lock()
	defer mu.Unlock()
	caseMark = len(cells)
}

// EndCase records PASS/FAIL for a suite case.
func EndCase(layer, name, status, errMsg string) {
	mu.Lock()
	defer mu.Unlock()
	n := len(cells) - caseMark
	if n < 0 {
		n = 0
	}
	cases = append(cases, CaseRow{Layer: layer, Name: name, Status: status, Err: errMsg, Cells: n})
}

// BeginLog clears logs/ and tees stdout into a single suite.txt.
func BeginLog() (restore func(), err error) {
	Reset()
	dir, err := logsDir()
	if err != nil {
		return nil, err
	}
	if err := os.RemoveAll(dir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "suite.txt")
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	r, w, err := os.Pipe()
	if err != nil {
		f.Close()
		return nil, err
	}
	mu.Lock()
	logFile = f
	logOld = os.Stdout
	logPipeW = w
	logDone = make(chan struct{})
	mu.Unlock()
	os.Stdout = w

	go func() {
		defer close(logDone)
		buf := make([]byte, 64*1024)
		var carry []byte
		for {
			n, readErr := r.Read(buf)
			if n > 0 {
				carry = append(carry, buf[:n]...)
				for {
					idx := bytes.Index(carry, []byte(flushToken))
					if idx < 0 {
						emit(carry)
						carry = carry[:0]
						break
					}
					if idx > 0 {
						emit(carry[:idx])
					}
					carry = carry[idx+len(flushToken):]
					mu.Lock()
					ch := flushCh
					flushCh = nil
					mu.Unlock()
					if ch != nil {
						close(ch)
					}
				}
			}
			if readErr != nil {
				if len(carry) > 0 {
					emit(carry)
				}
				_ = r.Close()
				return
			}
		}
	}()

	fmt.Fprintf(w, "=== w2a suite log ===\nfile=%s\ncleared=true\nstarted=%s\n\n",
		path, time.Now().Format(time.RFC3339))

	return func() {
		syncLog()
		_ = w.Close()
		<-logDone
		mu.Lock()
		if logFile != nil {
			_ = logFile.Close()
			logFile = nil
		}
		os.Stdout = logOld
		logOld = nil
		logPipeW = nil
		mu.Unlock()
	}, nil
}

func emit(p []byte) {
	if len(p) == 0 {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if logOld != nil {
		_, _ = logOld.Write(p)
	}
	if logFile != nil {
		_, _ = logFile.Write(p)
	}
}

func syncLog() {
	ch := make(chan struct{})
	mu.Lock()
	flushCh = ch
	mu.Unlock()
	fmt.Fprint(os.Stdout, flushToken)
	<-ch
}

// PrintReport prints the grand count + matrix/table (all layers).
func PrintReport() {
	syncLog()

	mu.Lock()
	elapsed := time.Since(started)
	cellsCopy := append([]Cell(nil), cells...)
	casesCopy := append([]CaseRow(nil), cases...)
	mu.Unlock()

	var ok, gap, fail, pass, caseFail int
	byLayer := map[string]struct{ OK, GAP, FAIL, Cases, CaseFail int }{}
	byOp := map[string]struct{ OK, GAP, FAIL int }{}

	for _, c := range cellsCopy {
		st := byLayer[c.Layer]
		op := byOp[c.Op]
		switch c.Status {
		case "OK":
			ok++
			st.OK++
			op.OK++
		case "GAP":
			gap++
			st.GAP++
			op.GAP++
		case "FAIL":
			fail++
			st.FAIL++
			op.FAIL++
		}
		byLayer[c.Layer] = st
		byOp[c.Op] = op
	}
	for _, c := range casesCopy {
		st := byLayer[c.Layer]
		st.Cases++
		if c.Status == "PASS" {
			pass++
		} else {
			caseFail++
			st.CaseFail++
		}
		byLayer[c.Layer] = st
	}

	totalCells := len(cellsCopy)
	fmt.Printf("\n")
	fmt.Printf("======== W2A SUITE REPORT ========\n")
	fmt.Printf("  matrix cells : %d   (OK %d  GAP %d  FAIL %d)\n", totalCells, ok, gap, fail)
	fmt.Printf("  suite cases  : %d   (PASS %d  FAIL %d)\n", len(casesCopy), pass, caseFail)
	fmt.Printf("  elapsed      : %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("==================================\n")

	fmt.Printf("\n  by layer\n")
	fmt.Printf("  %-10s %8s %8s %8s %8s %8s %8s\n", "layer", "cells", "OK", "GAP", "FAIL", "cases", "cFAIL")
	fmt.Printf("  %s\n", strings.Repeat("-", 66))
	for _, layer := range sortedKeys(byLayer) {
		st := byLayer[layer]
		fmt.Printf("  %-10s %8d %8d %8d %8d %8d %8d\n",
			layer, st.OK+st.GAP+st.FAIL, st.OK, st.GAP, st.FAIL, st.Cases, st.CaseFail)
	}

	fmt.Printf("\n  by op\n")
	fmt.Printf("  %-12s %8s %8s %8s %8s\n", "op", "cells", "OK", "GAP", "FAIL")
	fmt.Printf("  %s\n", strings.Repeat("-", 48))
	for _, op := range sortedKeysOp(byOp) {
		st := byOp[op]
		fmt.Printf("  %-12s %8d %8d %8d %8d\n", op, st.OK+st.GAP+st.FAIL, st.OK, st.GAP, st.FAIL)
	}

	fmt.Printf("\n  TOTAL TESTS (matrix cells) = %d\n", totalCells)
	if fail > 0 || caseFail > 0 {
		fmt.Printf("  RESULT: FAIL (%d cell fails, %d case fails)\n\n", fail, caseFail)
	} else {
		fmt.Printf("  RESULT: PASS\n\n")
	}
	syncLog()
}

func sortedKeys(m map[string]struct{ OK, GAP, FAIL, Cases, CaseFail int }) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	stringsSort(out)
	return out
}

func sortedKeysOp(m map[string]struct{ OK, GAP, FAIL int }) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	stringsSort(out)
	return out
}

func stringsSort(a []string) {
	for i := 0; i < len(a); i++ {
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[i] {
				a[i], a[j] = a[j], a[i]
			}
		}
	}
}

func logsDir() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("suites: no caller")
	}
	// .../w2a/suites/report.go → w2a/logs
	w2a := filepath.Clean(filepath.Join(filepath.Dir(file), ".."))
	return filepath.Join(w2a, "logs"), nil
}
