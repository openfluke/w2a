package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"sync/atomic"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/accel"
	"github.com/openfluke/welvet/runtime/dispatch"
)

var lastErr atomic.Value // string

func setLastErr(msg string) {
	lastErr.Store(msg)
}

func clearLastErr() { lastErr.Store("") }

//export WelvetLastError
func WelvetLastError() *C.char {
	v, _ := lastErr.Load().(string)
	if v == "" {
		return C.CString("")
	}
	return C.CString(v)
}

//export WelvetProbeBackends
func WelvetProbeBackends() *C.char {
	return jsonOut(accel.Probe())
}

//export WelvetEnableBackend
func WelvetEnableBackend(name *C.char) *C.char {
	b, err := accel.Parse(C.GoString(name))
	if err != nil {
		return errJSON(err.Error())
	}
	if err := accel.Ensure(b); err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{
		"status":  "ok",
		"backend": b.String(),
		"id":      C.GoString(name),
		"sync":    accel.NeedsSync(b),
	})
}

//export WelvetSetGridBackend
func WelvetSetGridBackend(handle C.longlong, name *C.char) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	b, err := accel.Parse(C.GoString(name))
	if err != nil {
		return errJSON(err.Error())
	}
	if err := accel.Ensure(b); err != nil {
		return errJSON(err.Error())
	}
	g.Exec.Backend = b
	g.Exec.BackendPinned = true
	for i := range g.Cells {
		if g.Cells[i].Op != nil {
			dispatch.SyncExec(g.Cells[i].Op, g.Exec)
		}
	}
	return jsonOut(map[string]any{
		"status":  "ok",
		"backend": b.String(),
		"sync":    accel.NeedsSync(b),
	})
}

//export WelvetPrepareGrid
func WelvetPrepareGrid(handle C.longlong) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	n := 0
	for i := range g.Cells {
		op := g.Cells[i].Op
		if op == nil {
			continue
		}
		switch g.Exec.Backend {
		case core.BackendQNN:
			if s, ok := op.(interface{ SyncToQNN() error }); ok {
				if err := s.SyncToQNN(); err != nil {
					return errJSON(err.Error())
				}
				n++
			}
		case core.BackendMLX:
			if s, ok := op.(interface{ SyncToMLX() error }); ok {
				if err := s.SyncToMLX(); err != nil {
					return errJSON(err.Error())
				}
				n++
			}
		default:
			// Routed cuda/metal/rocm/vulkan sync automatically after ops.
		}
	}
	return jsonOut(map[string]any{
		"status":  "ok",
		"backend": g.Exec.Backend.String(),
		"synced":  n,
		"needed":  accel.NeedsSync(g.Exec.Backend),
	})
}
