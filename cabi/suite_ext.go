package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	cnn1suite "github.com/openfluke/w2a/suites/cnn1"
	cnn2suite "github.com/openfluke/w2a/suites/cnn2"
	cnn3suite "github.com/openfluke/w2a/suites/cnn3"
	convt1suite "github.com/openfluke/w2a/suites/convt1"
	convt2suite "github.com/openfluke/w2a/suites/convt2"
	convt3suite "github.com/openfluke/w2a/suites/convt3"
	densesuite "github.com/openfluke/w2a/suites/dense"
	dnasuite "github.com/openfluke/w2a/suites/dna"
	donatesuite "github.com/openfluke/w2a/suites/donate"
	embeddingsuite "github.com/openfluke/w2a/suites/embedding"
	evosuite "github.com/openfluke/w2a/suites/evolution"
	fountainsuite "github.com/openfluke/w2a/suites/fountain"
	gdnsuite "github.com/openfluke/w2a/suites/gdn"
	hardwaresuite "github.com/openfluke/w2a/suites/hardware"
	helperssuite "github.com/openfluke/w2a/suites/helpers"
	kmeanssuite "github.com/openfluke/w2a/suites/kmeans"
	lnsuite "github.com/openfluke/w2a/suites/layernorm"
	lstmsuite "github.com/openfluke/w2a/suites/lstm"
	mambasuite "github.com/openfluke/w2a/suites/mamba"
	memorysuite "github.com/openfluke/w2a/suites/memory"
	metasuite "github.com/openfluke/w2a/suites/metacognition"
	mhasuite "github.com/openfluke/w2a/suites/mha"
	parallelsuite "github.com/openfluke/w2a/suites/parallel"
	residualsuite "github.com/openfluke/w2a/suites/residual"
	rmsnsuite "github.com/openfluke/w2a/suites/rmsnorm"
	rnnsuite "github.com/openfluke/w2a/suites/rnn"
	seedsuite "github.com/openfluke/w2a/suites/seed"
	sequentialsuite "github.com/openfluke/w2a/suites/sequential"
	serializationsuite "github.com/openfluke/w2a/suites/serialization"
	sevensuite "github.com/openfluke/w2a/suites/seven"
	softmaxsuite "github.com/openfluke/w2a/suites/softmax"
	stepsuite "github.com/openfluke/w2a/suites/step"
	swigsuite "github.com/openfluke/w2a/suites/swiglu"
	tweensuite "github.com/openfluke/w2a/suites/tween"
	weightssuite "github.com/openfluke/w2a/suites/weights"
)

type suiteEntry struct {
	Name string
	Run  func() error
}

func suiteCatalog() []suiteEntry {
	return []suiteEntry{
		{"dense", densesuite.RunAll},
		{"weights", weightssuite.RunAll},
		{"mha", mhasuite.RunAll},
		{"swiglu", swigsuite.RunAll},
		{"rmsnorm", rmsnsuite.RunAll},
		{"layernorm", lnsuite.RunAll},
		{"embedding", embeddingsuite.RunAll},
		{"softmax", softmaxsuite.RunAll},
		{"sequential", sequentialsuite.RunAll},
		{"residual", residualsuite.RunAll},
		{"cnn1", cnn1suite.RunAll},
		{"cnn2", cnn2suite.RunAll},
		{"cnn3", cnn3suite.RunAll},
		{"rnn", rnnsuite.RunAll},
		{"lstm", lstmsuite.RunAll},
		{"convt1", convt1suite.RunAll},
		{"convt2", convt2suite.RunAll},
		{"convt3", convt3suite.RunAll},
		{"gdn", gdnsuite.RunAll},
		{"mamba", mambasuite.RunAll},
		{"kmeans", kmeanssuite.RunAll},
		{"metacognition", metasuite.RunAll},
		{"parallel", parallelsuite.RunAll},
		{"step", stepsuite.RunAll},
		{"tween", tweensuite.RunAll},
		{"dna", dnasuite.RunAll},
		{"evolution", evosuite.RunAll},
		{"serialization", serializationsuite.RunAll},
		{"seed", seedsuite.RunAll},
		{"fountain", fountainsuite.RunAll},
		{"memory", memorysuite.RunAll},
		{"helpers", helperssuite.RunAll},
		{"donate", donatesuite.RunAll},
		{"hardware", hardwaresuite.RunAll},
		{"seven", sevensuite.RunAll},
	}
}

//export WelvetListSuiteCatalog
func WelvetListSuiteCatalog() *C.char {
	cat := suiteCatalog()
	names := make([]string, len(cat))
	for i, s := range cat {
		names[i] = s.Name
	}
	return jsonOut(names)
}

//export WelvetRunSuite
func WelvetRunSuite(name *C.char) *C.char {
	want := strings.ToLower(strings.TrimSpace(C.GoString(name)))
	suites.Reset()
	t0 := time.Now()
	for _, s := range suiteCatalog() {
		if s.Name != want {
			continue
		}
		err := s.Run()
		out := map[string]any{
			"suite":    s.Name,
			"ok":       err == nil,
			"elapsed_ms": time.Since(t0).Milliseconds(),
		}
		if err != nil {
			out["error"] = err.Error()
		}
		return jsonOut(out)
	}
	return errJSON(fmt.Sprintf("unknown suite %q", want))
}

//export WelvetRunAllSuites
func WelvetRunAllSuites(flagsJSON *C.char) *C.char {
	var flags struct {
		Only   []string `json:"only"`
		Skip   []string `json:"skip"`
		Quick  bool     `json:"quick"`
	}
	_ = json.Unmarshal([]byte(C.GoString(flagsJSON)), &flags)
	skip := map[string]bool{}
	for _, s := range flags.Skip {
		skip[strings.ToLower(s)] = true
	}
	only := map[string]bool{}
	for _, s := range flags.Only {
		only[strings.ToLower(s)] = true
	}
	if flags.Quick && len(only) == 0 {
		// quick: lightweight portable smoke (avoid dense/parallel RunAll mega matrices)
		only = map[string]bool{
			"seed": true, "serialization": true, "helpers": true, "memory": true, "fountain": true,
		}
	}

	suites.Reset()
	t0 := time.Now()
	results := make([]map[string]any, 0)
	okN, failN, skipN := 0, 0, 0
	for _, s := range suiteCatalog() {
		if len(only) > 0 && !only[s.Name] {
			continue
		}
		if skip[s.Name] {
			skipN++
			results = append(results, map[string]any{"suite": s.Name, "ok": true, "skipped": true})
			continue
		}
		st := time.Now()
		err := s.Run()
		row := map[string]any{
			"suite":      s.Name,
			"ok":         err == nil,
			"elapsed_ms": time.Since(st).Milliseconds(),
		}
		if err != nil {
			failN++
			row["error"] = err.Error()
		} else {
			okN++
		}
		results = append(results, row)
	}
	return jsonOut(map[string]any{
		"ok":         failN == 0,
		"passed":     okN,
		"failed":     failN,
		"skipped":    skipN,
		"elapsed_ms": time.Since(t0).Milliseconds(),
		"suites":     results,
	})
}
