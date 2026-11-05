# Welvet family — cross-repo test tally

Snapshot: **2026-11-06**. Sibling checkouts under `gs/`.

## How to read this

| Kind | Meaning |
|------|---------|
| **Leaf cases** | Named `[]Case{ Name }` checks in `w2a/suites` (real runnable matrix) |
| **`Test*`** | Go `func Test…` entrypoints |
| **Smoke bins** | Native `example_*_smoke` / `welvet_dense_fwd` CMake targets (not leaf matrices) |
| **Checklist `[x]`** | Sign-offs in `welvet.*/docs/TRAINING_CHECKLIST.md` — track **w2a `go_v_*`** coverage; **do not add** to leaf totals (same work, documented per backend) |

Native engines (**cpp / cuda / metal / vulkan**) do **not** ship their own gtest suites. Parity lives in **w2a** (`go_v_cpp`, `go_v_cuda`, …).

---

## Per-repo automated checks

| Repo | Leaf cases | `Test*` | Smoke / other | Notes |
|------|----------:|--------:|--------------:|-------|
| **[w2a](../README.md)** | **6163** | 470 (`tests/`) | TS 6 modules, CABI verify 1 | Owns almost all automated parity |
| [welvet](../../welvet) | 0 | **30** (excl. `third_party`) | 1 bench | Engine: mostly no tests; leftover in `nativecpp` / `nativeqnn` / sampling / apps |
| [welvet.cpp](../../welvet.cpp) | 0\* | 0 | **2** exes (`welvet_dense_fwd`, `example_dense_forward`) | Exercised via w2a `go_v_cpp` |
| [welvet.cuda](../../welvet.cuda) | 0\* | 0 | **2** exes (+ `release_smoke.py`) | Via w2a `go_v_cuda` |
| [welvet.metal](../../welvet.metal) | 0\* | 0 | **2** exes (+ `release_smoke.py`) | Via w2a `go_v_metal` |
| [welvet.vulkan](../../welvet.vulkan) | 0\* | 0 | **2** exes (+ `release_smoke.py`) | Via w2a `go_v_vulkan` |
| [welvet-to-rust_n_cpp](../../welvet-to-rust_n_cpp) | 0 | 7 | Rust `#[test]` **7**, scripts **2** | Port / overlay experiments |
| [welvet_sweep_2_snat](../../welvet_sweep_2_snat) | 0 | 10 | — | Sweep harness |
| [internal_welvet_apple_v_ollama](../../internal_welvet_apple_v_ollama) | 0 | 0 | **11** bench packages (25 `.go`) | Manual / online multilayer benches |

\*Native leaf coverage is counted once under **w2a** `go_v_*` (cpp 1873 + cuda 939 + metal 961 + rocm 939 + vulkan 939 + mlx 22 + qnn 16 = **5689**).

---

## Grand totals (automated — no checklist double-count)

| Bucket | Count |
|--------|------:|
| w2a suite leaf cases | **6163** |
| welvet engine `Test*` (no `third_party`) | **30** |
| welvet-to-rust_n_cpp (Go + Rust + scripts) | **16** |
| welvet_sweep_2_snat `Test*` | **10** |
| Native smoke executables (4 backends × 2) | **8** |
| **ALL-UP AUTOMATED** | **6227** |

w2a `Test*` (470) are **wrappers** around leaf cases — already included in 6163, so they are **not** added again.

---

## Checklist sign-offs (tracking only)

| Repo | `TRAINING_CHECKLIST.md` `[x]` | Open `[ ]` |
|------|------------------------------:|----------:|
| welvet.cpp | 2439 | 0 |
| welvet.cuda | 2240 | 16 |
| welvet.metal | 2237 | 0 |
| welvet.vulkan | 2245 | 1 |
| **Sum of `[x]` marks** | **9161** | |

These mark the same w2a native-parity work per backend. **All-up automated stays 6227**; checklist sum is documentation coverage, not +9161 extra tests.

---

## w2a leaf split (detail)

| Area | Leaf cases |
|------|----------:|
| Layer / system suites | 474 |
| `go_v_cpp` | 1873 |
| `go_v_metal` | 961 |
| `go_v_cuda` | 939 |
| `go_v_rocm` | 939 |
| `go_v_vulkan` | 939 |
| `go_v_mlx` | 22 |
| `go_v_qnn` | 16 |
| **w2a total** | **6163** |

Every leaf name: [TEST_INVENTORY.md](TEST_INVENTORY.md).
