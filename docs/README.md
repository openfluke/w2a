# Welvet / w2a documentation

## Split of responsibility

- **welvet** — AI engines only (layers, quant, SIMD, WebGPU, ENTITY, …)
- **w2a** — tests, CABI, and these docs

## Must-read

1. [../README.md](../README.md) — harness overview  
2. [../../README.md](../../README.md) — engine contract (no QAT, backends, folder map)  
3. [matrix.md](matrix.md) — dtype × quant × backend coverage tracker  
4. [dense.md](dense.md) — Dense layer status (first implementation)

## Legacy reference

`loom/poly` remains the behavioral oracle until Welvet reaches parity. Prefer re-implementing natively over copy-paste QAT paths.
