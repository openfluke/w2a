# Coverage matrix (tracker)

See **[welvet/README.md](../../README.md)** for the full v1 checklist and rules:

- **No tests in the engine** — only here in `w2a`
- **No fallbacks** — missing path / device → hard error
- **No hardcoded float32** — activations/grads are `Tensor[T]` for any `Numeric`
- **v1** only when every layer × dtype × quant × backend × fwd/bwd is real

## Dense (honest)

| Backend | What’s real today |
|---------|-------------------|
| CPU tiled | MatVec across dtype × quant (generic); keep hardening native paths |
| SIMD | Plan 9: FP32/`None`, Int8/`None`, Q4_0 fused — **everything else errors** |
| WebGPU | Device required; unbound / no adapter → **error** (no host fake) |

Run gaps: `w2a` → Dense → `[9] FULL aspirational matrix`.
