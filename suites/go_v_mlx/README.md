# go_v_mlx — Go vs Apple MLX GPU (forward-only)

Sibling of `go_v_qnn`. Validates **welvet.mlx** leaf forwards — **no training**.

## Protocol

1. Build identical weights / activations in Go  
2. **Transfer** — host materialize (timed)  
3. **Prep** — DensePrepare / array cast (`LastPhase` create)  
4. **Exec** — `mx::eval` window (`LastPhase` execute) — headline  
5. **E2E** = transfer + create + finalize + exec  
6. **Accuracy** — go↔mlx exact% / max|Δ| / RMSE where wired  

Sign-off checklist: [`welvet.mlx/docs/FORWARD_CHECKLIST.md`](../../../welvet.mlx/docs/FORWARD_CHECKLIST.md).

## Coverage (0.15 — 14 leaves)

| Layer | Caps dtypes | Notes |
|-------|-------------|-------|
| Dense | floats + ints + c64 | Session + Go Sync/Forward floats |
| Softmax / RMSNorm / LayerNorm | f32/f16/bf16 | Session + Go accuracy on floats |
| CNN2 / CNN3 / ConvT2 / ConvT3 | floats + ints | Session; float 1×1 vs Go where easy |
| LSTM / SwiGLU / MHA | f32/f16/bf16 | Session exec+finite |
| Embedding | floats + ints | Session; float gather vs Go |
| Residual | floats + ints + c64 | Session elementwise add |
| Sequential | f32 tower | Session Dense stack |

```bash
go test ./tests/go_v_mlx -run 'TestGoVMLX$|TestGoVMLXAll|TestGoVMLXDense' -v -count=1
```

## Gate

Skips unless `nativemlx.Available()` (cgo + darwin/arm64 + `libwelvet_mlx.dylib` + Metal).

```bash
../welvet.mlx/scripts/macos_build_w2a.sh
cd w2a
go test ./tests/go_v_mlx -v -count=1 -timeout 60m
```
