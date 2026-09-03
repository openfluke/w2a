// Welvet C-ABI (c-shared / c-archive) — Welvet 1.1.1 engine bridge for Python and native hosts.
package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import "unsafe"

//export FreeWelvetString
func FreeWelvetString(ptr *C.char) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

//export FreeLoomString
func FreeLoomString(ptr *C.char) { FreeWelvetString(ptr) }

//export WelvetEngineVersion
func WelvetEngineVersion() *C.char {
	return C.CString("1.1.1")
}

func main() {}
