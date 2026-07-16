package dense

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unicode"
)

const flushToken = "\x1eWELVET_LOG_FLUSH\x1e\n"

var (
	logMu     sync.Mutex
	caseFile  *os.File
	suiteFile *os.File
	logOldOut *os.File
	logPipeW  *os.File
	logDone   chan struct{}
	flushCh   chan struct{}
	logsRoot  string
)

// BeginLogging clears logs/dense/ and tees all stdout (fmt.Printf etc.) into
// suite.txt plus an optional per-case .txt while a case is active.
func BeginLogging() (restore func(), err error) {
	dir, err := denseLogsDir()
	if err != nil {
		return nil, err
	}
	if err := os.RemoveAll(dir); err != nil {
		return nil, fmt.Errorf("dense logs clear: %w", err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	logsRoot = dir

	suitePath := filepath.Join(dir, "suite.txt")
	sf, err := os.Create(suitePath)
	if err != nil {
		return nil, err
	}

	r, w, err := os.Pipe()
	if err != nil {
		sf.Close()
		return nil, err
	}

	logMu.Lock()
	suiteFile = sf
	caseFile = nil
	logOldOut = os.Stdout
	logPipeW = w
	logDone = make(chan struct{})
	flushCh = nil
	logMu.Unlock()

	os.Stdout = w

	go teeLoop(r)

	fmt.Fprintf(w, "=== dense suite log ===\n dir=%s\n cleared=true\n\n", dir)

	return func() {
		syncTee()
		_ = w.Close()
		<-logDone
		logMu.Lock()
		if caseFile != nil {
			_ = caseFile.Close()
			caseFile = nil
		}
		if suiteFile != nil {
			_ = suiteFile.Close()
			suiteFile = nil
		}
		os.Stdout = logOldOut
		logOldOut = nil
		logPipeW = nil
		logMu.Unlock()
	}, nil
}

func teeLoop(r *os.File) {
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
					emitLog(carry)
					carry = carry[:0]
					break
				}
				if idx > 0 {
					emitLog(carry[:idx])
				}
				carry = carry[idx+len(flushToken):]
				logMu.Lock()
				ch := flushCh
				flushCh = nil
				logMu.Unlock()
				if ch != nil {
					close(ch)
				}
			}
		}
		if readErr != nil {
			if len(carry) > 0 {
				emitLog(carry)
			}
			_ = r.Close()
			return
		}
	}
}

func emitLog(p []byte) {
	if len(p) == 0 {
		return
	}
	logMu.Lock()
	defer logMu.Unlock()
	if logOldOut != nil {
		_, _ = logOldOut.Write(p)
	}
	if suiteFile != nil {
		_, _ = suiteFile.Write(p)
	}
	if caseFile != nil {
		_, _ = caseFile.Write(p)
	}
}

func syncTee() {
	ch := make(chan struct{})
	logMu.Lock()
	flushCh = ch
	logMu.Unlock()
	fmt.Fprint(os.Stdout, flushToken)
	<-ch
}

// BeginCaseLog opens logs/dense/NN_sanitized_name.txt and includes it in the tee.
func BeginCaseLog(index int, name string) error {
	if logsRoot == "" {
		return fmt.Errorf("dense logs: BeginLogging not called")
	}
	syncTee()
	safe := sanitizeLogName(name)
	path := filepath.Join(logsRoot, fmt.Sprintf("%02d_%s.txt", index+1, safe))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	logMu.Lock()
	if caseFile != nil {
		_ = caseFile.Close()
	}
	caseFile = f
	logMu.Unlock()

	fmt.Fprintf(os.Stdout, "=== case %d: %s ===\n\n", index+1, name)
	syncTee()
	return nil
}

// EndCaseLog closes the current per-case log file.
func EndCaseLog() {
	syncTee()
	logMu.Lock()
	f := caseFile
	caseFile = nil
	logMu.Unlock()
	if f != nil {
		_, _ = f.WriteString("\n=== end case ===\n")
		_ = f.Close()
	}
}

func denseLogsDir() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("dense logs: no caller")
	}
	w2a := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	return filepath.Join(w2a, "logs", "dense"), nil
}

func sanitizeLogName(name string) string {
	var b strings.Builder
	prevUnderscore := false
	for _, r := range name {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(unicode.ToLower(r))
			prevUnderscore = false
		default:
			if !prevUnderscore && b.Len() > 0 {
				b.WriteByte('_')
				prevUnderscore = true
			}
		}
	}
	s := strings.Trim(b.String(), "_")
	if s == "" {
		s = "case"
	}
	if len(s) > 80 {
		s = s[:80]
	}
	return s
}
