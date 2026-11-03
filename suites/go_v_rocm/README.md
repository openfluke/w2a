# go_v_rocm — Go ↔ welvet.rocm determinism

Sibling of `go_v_cuda` / `go_v_metal` / `go_v_cpp`.

Engine: `../welvet.rocm` (HIP). Checklist:
`../welvet.rocm/docs/TRAINING_CHECKLIST.md` (mirror of cuda → board **0.79**).

**M0:** smoke via `welvet.rocm/scripts/linux_build_w2a.sh --smoke`.

As each leaf / NormalBP / TrainMode lands in the C++ engine, add suite files here
and `tests/go_v_rocm/*_test.go` using the same spectrum helpers as `go_v_cuda`
(`suites/spectrum.go`, `suites/determinism.go`).
