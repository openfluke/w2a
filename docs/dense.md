# Dense — w2a validation notes

Engine: `github.com/openfluke/welvet/layers/dense`  
Policy: see welvet README (no engine tests, no fallbacks, nothing hardcoded to float32 — `Forward[T]`/`Backward[T]`, v1 checklist).

DType matrix: **34** types (0–33). Suite includes CPU FormatNone smoke for all 34 and a **TIMED matrix** (fwd/bwd ns/op × CPU/SIMD/WebGPU).

**Fused Dense SIMD (k/IQ/Affine):** suite cases project packed codes → `Int8QS` + scales (`Ensure*SIMDCache`) and check CPU packed MatVec parity with `BackendSIMD` — no `F32Cache` inflate. See `suites/dense/fused_simd.go`.

```bash
cd welvet/w2a && go run .
# 1 Dense → pick a number
# or: go test ./tests/dense/ -v -run TestDenseSuite
```

Useful cases: SIMD fused k-cache / k/IQ/Affine parity, CPU matrix, gap census.
