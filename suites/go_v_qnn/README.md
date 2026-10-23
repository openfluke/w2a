# go_v_qnn — Go vs Hexagon QNN (README HTP matrix)

## Protocol

For every README HTP layer (f32):

1. Build weights / activations in Go
2. **Transfer** — time host weight materialize / Sync push (outside exec)
3. **Graph prep** — report **create** and **finalize** separately (`LastPhase` / `DensePrepare`)
4. **Exec** — time **graphExecute only** after weights are transferred (headline number)
5. **E2E** = transfer + create + finalize + exec
6. **Accuracy** — go↔qnn **exact%**, **max|Δ|**, **RMSE** (where a Go ref is wired)

Shapes are large enough that HTP can beat CPU on exec (Dense 64×1024×1024, Softmax 128×8192, …).

**Latest numbers:** [RESULTS.md](./RESULTS.md)

## Coverage

| Layer | Accuracy | Notes |
|-------|----------|-------|
| Dense | yes | Prepare then exec-only |
| Softmax / RMSNorm / LayerNorm | yes | Online graph; exec from phase split |
| CNN2 | yes | 1×1 Go ref |
| CNN3 / ConvT2 / ConvT3 / SwiGLU / MHA | phase timing | Go ref later |
| LSTM | — | HTP execute FAIL (compose) |
| Embedding / Residual | yes | gather / add |
| Sequential tower | go↔qnn | dim=128 depth=3 |

```bash
cd w2a
go test ./tests/go_v_qnn -v -count=1 -timeout 90m
```
