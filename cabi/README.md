# CABI harness (w2a)

Shared C ABI and language bindings for Welvet engines will live here
(mirroring `loom/welvet/cabi` + platform libs), **not** inside engine packages.

Planned:

- `include/welvet.h` — stable C surface
- `go/` — `c-shared` build wrappers
- `dart/` `python/` `typescript/` — thin clients
- CI scripts that consume `github.com/openfluke/welvet` only as a library
