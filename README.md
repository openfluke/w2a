# w2a — Welvet validation, CABI, docs

Companion to **[openfluke/welvet](https://github.com/openfluke/welvet)**.

| Path | Purpose |
|------|---------|
| `docs/` | Architecture notes, dtype×k-quant matrices, migration from `loom/poly` |
| `tests/` | Go parity / smoke / matrix suites against `github.com/openfluke/welvet` |
| `cabi/` | C ABI + language bindings harness (Dart/Python/TS/… later) |
| `scripts/` | Generators / CI helpers |

**Rules:** engine code stays in `welvet/` (no `*_test.go`, no fallbacks, no hardcoded float32 tensors).  
Everything that *proves* or *wraps* the engine lives here. See `../README.md` for the v1 checklist.

## Interactive menu (recommended)

```bash
cd welvet/w2a
go run .
```

Then pick a number:

- `[0]` run all suites  
- `[1]` Dense → sub-menu (`0` = all dense cases, `1..n` = one case, `b` = back)  
- `[q]` quit  

Suites live in `suites/<layer>/` so the menu and `go test` share the same checks.

## Plain `go test`

```bash
cd welvet/w2a
go test ./tests/dense/ -v          # Dense only
go test ./...                      # everything
```

Local module replace: `replace github.com/openfluke/welvet => ../`
