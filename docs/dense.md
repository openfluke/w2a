# Dense — w2a validation notes

Engine: `github.com/openfluke/welvet/dense`  
Policy: see welvet README (no engine tests, no fallbacks, nothing hardcoded to float32 — `Forward[T]`/`Backward[T]`, v1 checklist).

DType matrix: **34** types (0–33). Suite includes CPU FormatNone smoke for all 34 and a **TIMED matrix** (fwd/bwd ns/op × CPU/SIMD/WebGPU).

```bash
cd welvet/w2a && go run .
# 1 Dense → pick a number
```

Useful cases: CPU matrix `[7]`, SIMD implemented `[8]`, gap census `[9]`.
