# w2a — Welvet validation, CABI, bindings, docs

Companion to **[openfluke/welvet](https://github.com/openfluke/welvet)**.

**Rules:** engine code stays in `welvet/` (no `*_test.go`, no silent fallbacks, no hardcoded float32 tensors).  
Everything that *proves* or *wraps* the engine lives here.

Keep `w2a/` and `welvet/` as sibling checkouts. Local module replace:

```
replace github.com/openfluke/welvet => ../welvet
```

GPU / native parity suites skip or fail closed when the sibling backend binary is missing (`welvet.cpp`, `welvet.vulkan`, CUDA/Metal/ROCm toolchains, etc.).

---

## What’s in this repo

| Path | Purpose |
|------|---------|
| `tests/` | Go `*_test.go` packages — layer smoke + **Go ↔ native** parity (`go_v_*`) |
| `suites/` | Shared case logic for the interactive menu (`go run .`) and `go test` |
| `main.go` | Interactive suite menu (37 suites) |
| `cabi/` | C ABI (`c-shared` / archives) + `cabi_verify` |
| `python/` | ctypes bindings over the C ABI |
| `typescript/` + `wasm/` | npm `@openfluke/welvet` + Go→WASM |
| `tdeploy/` | Node / Bun / React deploy smoke |
| `docs/` | Coverage notes + **[full leaf test inventory](docs/TEST_INVENTORY.md)** |
| `scripts/` | Cross-OS `go_v_cpp` snapshot helpers |
| `logs/` | Checked-in / local suite snapshots (e.g. Linux `go_v_cpp`) |
| `examples/` | Small demos (e.g. dense dtype heatmap) |

---

## Inventory tally (filesystem, 2026-11-06)

**Leaf cases** = named entries in `suites/**` `[]Case{ Name: "…" }` (what `go run .` / `Cases()` actually runs).  
**`Test*`** = Go test entrypoints in `tests/**` — many of those call a whole `Cases()` list.

- Every leaf + every `Test*` name: **[docs/TEST_INVENTORY.md](docs/TEST_INVENTORY.md)**  
- Every Welvet sibling repo rollup: **[docs/WELVET_FAMILY_TEST_TALLY.md](docs/WELVET_FAMILY_TEST_TALLY.md)**

### Welvet family — all repos → all up

Native **cpp / cuda / metal / vulkan** do not own separate gtest matrices; their parity is the **w2a `go_v_*` leaf cases**. Checklist `[x]` marks in those repos track the same work — **not** added again below.

| Repo | Automated checks | What they are |
|------|-----------------:|---------------|
| **w2a** | **6163** | Suite leaf cases (layer 474 + `go_v_*` 5689) |
| **welvet** | **30** | Engine `Test*` (excl. `third_party`) |
| **welvet.cpp** | **2** | Smoke / CLI exes (`welvet_dense_fwd`, `example_dense_forward`) |
| **welvet.cuda** | **2** | Smoke / CLI exes |
| **welvet.metal** | **2** | Smoke / CLI exes |
| **welvet.vulkan** | **2** | Smoke / CLI exes |
| **welvet-to-rust_n_cpp** | **16** | 7 Go + 7 Rust + 2 scripts |
| **welvet_sweep_2_snat** | **10** | Sweep `Test*` |
| **internal_welvet_apple_v_ollama** | **11** pkgs | Online multilayer benches (not leaf cases) |
| | | |
| **ALL-UP AUTOMATED** | **6227** | w2a 6163 + welvet 30 + rust/sweep 26 + native smokes 8 |

w2a’s **470 `Test*`** are wrappers around the 6163 leaves — **not** counted twice.

Tracking only (do **not** add to 6227): TRAINING_CHECKLIST `[x]` ≈ cpp 2439 + cuda 2240 + metal 2237 + vulkan 2245 = **9161** sign-off marks.

### w2a-only grand totals

| Metric | Count |
|--------|------:|
| **Suite leaf cases** (`Name:` in `[]*Case{}`) | **6163** |
| — layer / system suites | **474** |
| — `go_v_*` native parity | **5689** |
| Go `func Test*` entrypoints | **470** |
| `func Benchmark*` | **4** |
| `tests/` packages | **47** |
| `*_test.go` files | **366** |
| `suites/` packages | **43** |
| Interactive menu suites | **37** |
| Native parity backends | **7** |
| TypeScript test modules | **6** |
| CABI C verify | **1** |

### Layer / system leaf cases (474)

| Suite | Leaf cases | Suite | Leaf cases |
|-------|----------:|-------|----------:|
| `parallel` | 36 | `convt1` | 14 |
| `dense` | 25 | `convt2` | 14 |
| `mha` | 23 | `convt3` | 14 |
| `softmax` | 23 | `kmeans` | 14 |
| `cnn1` | 18 | `mamba` | 14 |
| `layernorm` | 18 | `metacognition` | 14 |
| `rmsnorm` | 18 | `step` | 13 |
| `swiglu` | 18 | `helpers` | 11 |
| `embedding` | 17 | `tween` | 10 |
| `gdn` | 17 | `serialization` | 9 |
| `lstm` | 17 | `seven` | 9 |
| `rnn` | 17 | `seed` | 8 |
| `cnn2` | 15 | `dna` | 6 |
| `cnn3` | 15 | `evolution` | 6 |
| `residual` | 15 | `memory` | 3 |
| `sequential` | 15 | `donate` / `fountain` / `hardware` / `weights` | 2 each |

Example — `suites/dense` has **25** leaves (not 1 suite): forward/backward smokes, SIMD fused k/IQ/Affine, timed dtype×quant matrices, train volumetric grids, gap census, … (full list in the inventory).

### Native parity leaf cases (5689)

| Backend | Leaf cases | Unique names | `Test*` entrypoints |
|---------|----------:|-------------:|--------------------:|
| `go_v_cpp` | **1873** | 1637 | 72 |
| `go_v_metal` | **961** | 883 | 87 |
| `go_v_cuda` | **939** | 863 | 79 |
| `go_v_rocm` | **939** | 863 | 61 |
| `go_v_vulkan` | **939** | 863 | 61 |
| `go_v_mlx` | **22** | 22 | 4 |
| `go_v_qnn` | **16** | 16 | 2 |
| **Total** | **5689** | | **366** |

Each `TestGoVVulkanCNN`, `TestGoVCppDenseTrain`, etc. expands into many named cases (exact-bits, FormatNone×34 spectrum, train volumes, tween modes, …).

### `tests/` package table (`Test*` wrappers)

| Package | `Test*` | Leaf cases (via suite) | Role |
|---------|--------:|-----------------------:|------|
| `cnn1`…`cnn3` | 1 each | 18 / 15 / 15 | Conv wrappers |
| `convt1`…`convt3` | 1 each | 14 each | ConvTranspose |
| `dense` | 3 | 25 (+ benches) | Dense + GEMV bench + proxy |
| `densenet` | 2 | — | Volumetric DenseNet-style |
| `dna` | 3 | 6 | DNA census |
| `embedding` | 1 | 17 | Embedding |
| `entity` | 9 | — | ENTITY + cameral (`Test*` are the leaves) |
| `evolution` | 2 | 6 | Clone / splice |
| `fusedgpu` | 3 | — | Fused-GPU pack / limits |
| `gdn` | 1 | 17 | Gated DeltaNet |
| `go_v_cpp` | 72 | **1873** | Go ↔ welvet.cpp |
| `go_v_cuda` | 79 | **939** | Go ↔ CUDA |
| `go_v_metal` | 87 | **961** | Go ↔ Metal |
| `go_v_mlx` | 4 | **22** | Go ↔ MLX |
| `go_v_qnn` | 2 | **16** | Go ↔ QNN |
| `go_v_rocm` | 61 | **939** | Go ↔ ROCm |
| `go_v_vulkan` | 61 | **939** | Go ↔ Vulkan |
| `hf` | 3 | — | HF inspect / maxseq |
| `kmeans` | 1 | 14 | Soft K-Means |
| `layernorm` / `rmsnorm` / `swiglu` | 1 each | 18 each | Norm / FFN |
| `lstm` / `rnn` | 1 each | 17 each | Recurrent |
| `lucy` | 3 | — | Lucy-style |
| `mamba` | 1 | 14 | SSM |
| `metacognition` | 2 | 14 | Metacognition |
| `mha` | 1 | 23 | MHA / RoPE / GQA |
| `parallel` | 33 | 36 suite + many credit `Test*` | MoE Parallel + credit perms |
| `quant` | 1 | — | BinaryG128 (+ 1 bench) |
| `residual` / `sequential` | 2 each | 15 each | Compose |
| `sampling` | 4 | — | Sampling helpers |
| `serialization` | 1 | 9 | JSON + ENTITY |
| `seven` | 1 | 9 | Lucy-[7] |
| `simd` | 1 | — | SAXPY smoke |
| `softmax` | 1 | 23 | Softmax |
| `step` | 2 | 13 | Step mesh |
| `tanhi` / `telemetry` | 1 each | — | Helpers |
| `transformer` | 2 | — | Transformer glue |
| `tween` | 2 | 10 | StepTween |
| `wav2vec2` | 3 | — | Wav2Vec2-related |
| `weights` | 1 | 2 | ApplySGD |
| **TOTAL** | **470** | **~6163** suite leaves | |

---

## Native parity backends (`go_v_*`)

| Backend | Leaf cases | `Test*` | Test files | Sibling / tool | Menu? |
|---------|----------:|--------:|-----------:|----------------|-------|
| `go_v_cpp` | **1873** | 72 | 72 | `../welvet.cpp/build/welvet_dense_fwd` | ✅ `Go_v_Cpp` |
| `go_v_cuda` | **939** | 79 | 59 | CUDA toolkit + Welvet CUDA paths | `go test` only |
| `go_v_metal` | **961** | 87 | 66 | Apple Metal / `welvet` Metal | `go test` only |
| `go_v_rocm` | **939** | 61 | 56 | ROCm | `go test` only |
| `go_v_vulkan` | **939** | 61 | 56 | `../welvet.vulkan` (`WELVET_VULKAN_DENSE_FWD`) | `go test` only |
| `go_v_mlx` | **22** | 4 | 1 | Apple MLX | `go test` only |
| `go_v_qnn` | **16** | 2 | 1 | Hexagon QNN (windows/arm64) | ✅ `Go_v_QNN` |
| **Sum** | **5689** | **366** | | | |

Every leaf name is listed under `suites/go_v_*/…` in [docs/TEST_INVENTORY.md](docs/TEST_INVENTORY.md).

### Shared layer coverage (test file present)

Core layers appear on cpp / cuda / metal / rocm / vulkan. C++ also carries extra tween-split / freeze / shadow train leaves; GPU backends add `cameral_train`, `normalbp_all`, `proxy_train`, mesh modes, etc.

| Layer family | cpp | cuda | metal | rocm | vulkan |
|--------------|:---:|:----:|:-----:|:----:|:------:|
| cnn / cnn_train | ✅ | ✅ | ✅ | ✅ | ✅ |
| convt / convt_train | ✅ | ✅ | ✅ | ✅ | ✅ |
| dense_train | ✅ | ✅ | ✅ | ✅ | ✅ |
| embedding (+ train) | ✅ | ✅ | ✅ | ✅ | ✅ |
| gdn (+ train) | ✅ | ✅ | ✅ | ✅ | ✅ |
| kmeans (+ train†) | ✅ | ✅ | ✅† | ✅ | ✅ |
| layernorm / lstm / rnn (+ train) | ✅ | ✅ | ✅ | ✅ | ✅ |
| mamba / mha / swiglu / softmax / rmsnorm (+ train) | ✅ | ✅ | ✅ | ✅ | ✅ |
| sequential / residual / parallel | ✅ | ✅ | ✅ | ✅ | ✅ |
| metacognition (+ train) | ✅ | ✅ | ✅ | ✅ | ✅ |
| tween* / steptween* train families | ✅ | ✅ | ✅ | ✅ | ✅ |
| cameral_train / normalbp_all / proxy_train | — | ✅ | ✅ | ✅ | ✅ |
| mesh_split_modes / meshbp_train | — | ✅ | ✅ | ✅ | ✅ |
| Extra cpp-only tween-split / freeze / shadow | ✅ | — | — | — | — |

† Metal has `kmeans` smoke; `kmeans_train` file present on cuda/rocm/vulkan/cpp.

See package READMEs under `suites/go_v_*/` for env vars and build scripts.

---

## Interactive menu suites (`go run .`)

37 suites in `main.go` — same logic as `suites/<name>/`, usually wrapped by `tests/<name>/`.

| # | Suite | What it checks |
|---|-------|----------------|
| 1 | Go_v_Cpp | Exact 1:1 Go Welvet vs `welvet.cpp` (+ speed); needs `../welvet.cpp/build` |
| 2 | Go_v_QNN | Go vs native vs Hexagon QNN timing (QNN-capable layers; windows/arm64) |
| 3 | Dense | 34 dtypes, timed matrix, gap census |
| 4 | Weights | Native ApplySGD — FormatNone × dtypes, no retained f32 scratch |
| 5 | MHA | Causal + RoPE + GQA; FormatNone×34 + quants × backends + train grids |
| 6 | SwiGLU | SiLU-gated FFN; dtype/quant/backend/train grids |
| 7 | RMSNorm | γ-scale RMSNorm grids |
| 8 | LayerNorm | γ+β LayerNorm grids |
| 9–11 | CNN1 / CNN2 / CNN3 | Conv1d/2d/3d im2col→Dense grids |
| 12–13 | RNN / LSTM | Recurrent grids |
| 14 | Embedding | Token gather/scatter grids |
| 15 | Softmax | Weightless last-axis Softmax |
| 16 | Sequential | Dense→Dense compose |
| 17 | Residual | `y = F(x)+x` |
| 18–20 | ConvT1 / ConvT2 / ConvT3 | ConvTranspose + SIMD + gap census |
| 21 | Parallel | MoE-style Parallel concat/add/avg/filter |
| 22 | KMeans | Soft K-Means centers |
| 23 | Mamba | Selective SSM (`seqmix.KindSSM`) |
| 24 | Metacognition | Heuristic wrapper (no QAT morph) |
| 25 | GDN | Gated DeltaNet decode + seq Forward |
| 26 | DNA | All layers × FormatNone×34 + quants×f32 + full census |
| 27 | Evolution | Clone+splice census |
| 28 | Tween | StepTween SIMD DotTile + layer×dtype×quant×backend census |
| 29 | Step | Volumetric step mesh Forward/Backward/ApplyTween |
| 30 | Seed | SeedFrom/He manifests (loom bit-compat) |
| 31 | Fountain | Luby Transform peel + RecoverWeightBlobs |
| 32 | Donate | Networked compute offload TCP protocol |
| 33 | Hardware | SystemAudit OS/CPU/RAM/GPU/disk/network |
| 34 | Memory | Footprint / history / ReleaseTransient |
| 35 | Serialization | JSON+ENTITY dtype×quant create/train/reload |
| 36 | Helpers | Grafting, templates, ensemble, clustering, observer, … |
| 37 | Seven | Lucy-[7]-style SC↔MC, det, train, ENTITY, S/M/L tiers |

Menu: `[0]` = run all · numbered = one suite · many suites have a sub-menu for individual cases · `[q]` quit.

**Not in the interactive menu (run via `go test`):** `go_v_cuda`, `go_v_metal`, `go_v_rocm`, `go_v_vulkan`, `go_v_mlx`, plus host-only packages like `entity`, `fusedgpu`, `hf`, `lucy`, `quant`, `sampling`, `simd`, `tanhi`, `telemetry`, `transformer`, `wav2vec2`.

---

## How to run

### Interactive (recommended)

```bash
cd w2a
go run .
```

### Plain `go test`

```bash
cd w2a
go test ./tests/dense/ -v
go test ./tests/seven/ -v
go test ./tests/go_v_cpp/ -v          # needs welvet.cpp build
go test ./tests/go_v_vulkan/ -v       # needs welvet.vulkan build + env
go test ./tests/go_v_cuda/ -v         # needs CUDA
go test ./...                         # everything (GPU pkgs skip/fail without tools)
```

### Go vs C++ parity

```bash
../welvet.cpp/scripts/build.sh
cd w2a
go test ./tests/go_v_cpp/ -v
# optional: WELVET_DENSE_FWD=/path/to/welvet_dense_fwd
# snapshots: scripts/run_go_v_cpp_linux_snapshot.sh
```

Compares Dense FormatNone×**all 34 dtypes** on the Welvet determinism spectrum  
(`broken → ballpark → loose → industry → strict → exact_bit`, aim = **100% exact bits**),  
plus per-dtype Go vs C++ timings. See `suites/spectrum.go`.

### Go vs Vulkan

```bash
cd ../welvet.vulkan && ./scripts/linux_build_w2a.sh --dense-fwd   # etc.
cd ../w2a
export WELVET_VULKAN_DENSE_FWD=../welvet.vulkan/build/welvet_dense_fwd
go test ./tests/go_v_vulkan/ -v -count=1
```

### Bindings

```bash
# CABI
cd cabi && ./build_macos.sh          # or internal/build/build_linux.sh --test

# Python (after CABI dist)
cd python && pip install -e . && python -m welvet.cabi_verify

# TypeScript / WASM
cd typescript && npm test            # see typescript/README.md
```

---

## Docs & coverage trackers

| Doc | Content |
|-----|---------|
| [docs/README.md](docs/README.md) | Doc index / responsibility split |
| [docs/matrix.md](docs/matrix.md) | dtype × quant × backend tracker |
| [docs/dense.md](docs/dense.md) | Dense layer status |
| `suites/go_v_*/README.md` | Per-backend parity notes |
| Sibling `welvet/README.md` | Engine contract & v1 checklist |

---

## Release

Attaches current `logs/suite.txt` + `logs/suite.pdf` to a GitHub Release tagged with the Welvet scorecard version from `../welvet/README.md`.

```bash
cd w2a
go run .            # run suites → logs/suite.txt (+ export PDF if needed)
./release.sh        # commit source, push, upload suite txt/pdf
./release.sh --dry-run
```

Needs `gh auth login` or `GITHUB_TOKEN`. `logs/` and `dist/` stay mostly gitignored (sample snapshots may remain).

---

## Recomputing this tally

Leaf cases = `Name:` fields inside `[]*Case{…}` under `suites/`.  
`Test*` = `^func Test` under `tests/`.

Regenerate the full name dump:

```bash
cd w2a
# (re-run the inventory script used to build docs/TEST_INVENTORY.md, or)
rg -n 'Name:\s*"' suites/ --glob '!*_test.go' | wc -l
rg -c '^func Test' tests --glob '*_test.go'
```

Refresh README tables + `docs/TEST_INVENTORY.md` when cases are added.
