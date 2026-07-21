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
| SIMD | Fused for all 20 quants (classic Q* + k/IQ group Dot* + Affine code-dot + BitNet); FormatNone×34 |
| WebGPU | Device required; unbound / no adapter → **error** (no host fake) |

Run gaps: `w2a` → Dense → gap census. Fused k/IQ/Affine parity lives in Dense suite (`fused_simd.go`).
