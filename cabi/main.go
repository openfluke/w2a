// Welvet C-ABI (c-shared / c-archive) — Welvet 1.1.2 engine bridge for Python and native hosts.
// 1.1.2: create_grid / list_backends select cuda·metal·rocm·vulkan·native_cpu·qnn·mlx via runtime/accel.
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
	return C.CString("1.1.2")
}

func main() {}
