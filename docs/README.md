# Welvet / w2a documentation

## Split of responsibility

- **welvet** — AI engines only (layers, quant, SIMD, WebGPU, ENTITY, …)
- **w2a** — tests, CABI, language bindings, suite menu, and these docs

## Must-read

1. **[../README.md](../README.md)** — harness overview, tally tables, how to run everything  
2. **[WELVET_FAMILY_TEST_TALLY.md](WELVET_FAMILY_TEST_TALLY.md)** — **every Welvet repo → all-up total (6227 automated)**  
3. **[TEST_INVENTORY.md](TEST_INVENTORY.md)** — every w2a leaf case name + every `func Test*`  
4. **[../../welvet/README.md](../../welvet/README.md)** (sibling) — engine contract (no QAT, backends, folder map)  
5. [matrix.md](matrix.md) — dtype × quant × backend coverage tracker  
6. [dense.md](dense.md) — Dense layer status (first implementation)

## Test inventory (summary)

| Bucket | Count |
|--------|------:|
| **ALL-UP AUTOMATED (Welvet family)** | **6227** |
| w2a suite leaf cases | **6163** |
| — layer / system | 474 |
| — `go_v_*` | 5689 |
| Go `func Test*` entrypoints (w2a wrappers) | 470 |
| Interactive menu suites | 37 |

Backends: **cpp, cuda, metal, rocm, vulkan, mlx, qnn**. A single `Test*` usually expands into many named leaf cases.

## Legacy reference

`loom/poly` remains a behavioral oracle until Welvet reaches full parity. Prefer re-implementing natively over copy-paste QAT paths.
