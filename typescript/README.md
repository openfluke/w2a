# @openfluke/welvet

**Welvet 1.1.1** — isomorphic TypeScript / WebAssembly bindings for the [Welvet](https://github.com/openfluke/welvet) engine: volumetric grids, layers, dtypes/quants, cameral TrainModes, CamSync/CamKit, Tanhi, DNA, `.entity` checkpoints, seed / fountain / helpers.

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Engine:** [Welvet](https://github.com/openfluke/welvet) · **Book / examples:** [openfluke/example](https://github.com/openfluke/example) · **Feature book:** [openfluke.github.io/welvet](https://openfluke.github.io/welvet/)

## What this package is

`@openfluke/welvet` ships a Go → WASM build of Welvet (`apps/w2a`). Your Node, Bun, or browser code talks to the **same engine** as native Go — layers, training, DNA, cameral — without rewriting math in JS.

| Binding | Best for |
|--------|----------|
| **This package (WASM)** | Browser, Node.js, Bun — one `main.wasm` + `wasm_exec.js` |
| **Go `welvet/`** | Reference, Lucy / w2a harness, max CPU parallelism |

This **replaces** the Loom-era npm line (`@openfluke/welvet@0.80.0`). Version **1.1.1** is Welvet.

## Breaking change from 0.80 (Loom)

| 0.80 (Loom) | 1.1.1 (Welvet) |
|-------------|----------------|
| `LOOM_ENGINE_VERSION` (`"0.80.0"`) | `WELVET_ENGINE_VERSION` (`"1.1.1"`) |
| `createLoomNetwork` / `createNetwork` (JSON Lucy layout) | `createGrid` / `createWelvetGrid` + `placeDense` / … |
| `forwardPolymorphic` | `grid.forward(data, shapeJson?)` |
| `trainNetwork` / `net.train` | `grid.trainSGD` / `trainTween` / cameral `trainStackMSE` |
| — | Cameral: bicameral, hemispheres, TrainModes, CamSync, CamKit, Tanhi |

Loom aliases (`createNetwork`, `LOOM_ENGINE_VERSION`) still exist for a soft migration, but the **place + forward** API is Welvet’s grid model.

## Install

```bash
npm install @openfluke/welvet@1.1.1
# or
bun add @openfluke/welvet@1.1.1
```

Published tarball includes prebuilt `dist/main.wasm` (~14 MB). No Go toolchain needed for consumers.

### Build from source (monorepo)

```bash
cd apps/w2a/typescript
npm install
npm run build:all    # wasm/build.sh → assets → tsc → dist/
npm test             # smoke + coverage + layers + cameral + systems + quick matrix
```

## Quick start (Node / Bun)

```ts
import {
  init,
  createGrid,
  assertEngineVersion,
  DType,
} from "@openfluke/welvet";

await init();              // loads dist/main.wasm
assertEngineVersion();     // WASM must match package 1.1.1

const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });

// place* APIs take a JSON string (Go syscall/js)
g.placeDense(
  JSON.stringify({
    in: 4,
    out: 4,
    act: "relu",
    dtype: DType.FLOAT32, // 1
  }),
);

const x = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const fwd = g.forward(x);
if (fwd.error) throw new Error(fwd.error);
console.log("out", fwd.output);

const step = g.trainSGD(x, new Float32Array([0, 1, 0, 0]), 0.05);
console.log("loss", step.loss);

const entity = g.serializeEntity(); // Uint8Array — .entity checkpoint
g.free();
```

### Browser

Copy `node_modules/@openfluke/welvet/dist/main.wasm` (and optionally `wasm_exec.js`) into your static assets, or point `initBrowser` at a URL:

```ts
import { initBrowser, createGrid } from "@openfluke/welvet";

await initBrowser("/assets/main.wasm");
const g = createGrid();
g.placeDense(JSON.stringify({ in: 8, out: 4, act: "relu", dtype: 1 }));
console.log(g.forward(new Float32Array(8).fill(0.1)));
```

Minimal HTML demo (after `npm run build:all`):

```bash
npm run serve
# → http://localhost:3000/  (matrix.html, cabi_verify.html, …)
# local example page:
# → open examples/browser.html via the same static root, or see below
```

## Cameral (TrainModes)

```ts
import {
  init,
  createBicameral,
  createHemispheres,
  listNamedTrainModes,
} from "@openfluke/welvet";

await init();
const modes = listNamedTrainModes(); // NormalBP, StepBP, Tween, … (~31)

const stack = createBicameral({ in: 4, hidden: 4, out: 4 });
const x = new Float32Array(4).fill(0.1);
const y = new Float32Array([1, 0, 0, 0]);
console.log(stack.trainStackMSE(x, y, modes[0], 0.05));

const hem = createHemispheres({ dim: 4, n: 2, combine: "add" });
hem.setBranchModes(JSON.stringify([modes[0], modes[0]]));
hem.setCamSync(JSON.stringify({ Enabled: true, Alpha: 1 }));
hem.setCamKit(JSON.stringify({ ShadowCoef: 1, DNAReg: 0, SurpriseThresh: 0 }));
console.log(hem.trainMSE(x, y, modes[0], 0.05));
```

## Layers, dtypes, stores

```ts
import { createGrid, createStore, listDTypes, listFormats, listLayerTypes } from "@openfluke/welvet";

console.log(listLayerTypes());
console.log(listDTypes().length, listFormats().length);

const g = createGrid();
g.placeMHA(JSON.stringify({ DModel: 8, NumHeads: 2, SeqLen: 4, dtype: 1 }));
const seq = new Float32Array(32).fill(0.05);
console.log(g.forward(seq, JSON.stringify([1, 4, 8])));

const store = createStore(8, 8, 1, 0); // rows, cols, dtype, format
store.applySGD(new Float64Array(64).fill(0.01), 0.1);
console.log(store.flattenF32().length);
```

Useful `place*` methods on the grid handle: `placeDense`, `placeMHA`, `placeSwiGLU`, `placeRMSNorm`, `placeLayerNorm`, `placeEmbedding`, `placeSoftmax`, `placeSequential`, `placeResidual`, `placeCNN1`…`3`, `placeRNN` / `placeLSTM`, `placeConvT1`…`3`, `placeGDN`, `placeMamba`, `placeKMeans`, `placeParallel`, `placeMetacognition`, `placeStack`.

## Persistence (`.entity`)

```ts
const bytes = g.serializeEntity();           // Uint8Array
const back = DeserializeGrid(bytes);         // WASM global after init()
back.free();
```

JSON DNA / blueprint: `g.extractDNA()`, `g.extractBlueprint()`.

## Seed & portable helpers

```ts
import { seedFrom } from "@openfluke/welvet";

const s = seedFrom("welvet", 42, true); // deterministic string seed
```

After `init()`, additional WASM globals are available (fountain LT, memory fingerprint, grafting, Lucy scores, sampling, …). See the [example book WASM runners](https://github.com/openfluke/example) under `welvet/*/npm` and `_wasm/demos.mjs`.

## Examples in this package

| Path | What |
|------|------|
| [`examples/consumer_demo.ts`](examples/consumer_demo.ts) | npm consumer smoke: init → place → forward → train → entity |
| [`examples/run-example-smoke.ts`](examples/run-example-smoke.ts) | dense + bicameral + hemispheres / CamSync |
| [`examples/browser.html`](examples/browser.html) | one-file browser demo (serve `dist/` + this file) |

```bash
npm run test:consumer
npm run example
```

Full chapter mirrors (Go + npm + HTML for every book page): clone / use [`openfluke/example`](https://github.com/openfluke/example) with `WELVET_TS` pointing at this package tree.

## Tests & coverage matrix

| Command | Scope |
|---------|--------|
| `npm test` | smoke + coverage + layers + cameral + systems + **quick** matrix |
| `npm run test:smoke` | version, dense, trainSGD, entity |
| `npm run test:consumer` | README / npm install path |
| `npm run test:all:quick` | small accessibility matrix |
| `npm run test:all` | **mega** (~200k+ cells) — CI / release confidence |

Native-only (explicit SKIP in matrix): host SIMD Plan‑9, WebGPU device benches, donate TCP, hardware audit, FS Load\* paths.

```bash
npm run serve   # http://localhost:3000/matrix.html
```

## Publish (maintainers)

```bash
cd apps/w2a/typescript
npm login                 # once
bash publish.sh           # build:all → smoke+consumer → confirm → npm publish
# dry run only:
bash publish.sh --dry-run
```

`publish.sh` refuses to publish if `dist/main.wasm` is missing or `welvetEngineVersion()` ≠ `1.1.1`.

## Version alignment

| Component | Version |
|-----------|---------|
| **Welvet engine** | **1.1.1** |
| **npm `@openfluke/welvet`** | **1.1.1** (this package) |
| *Previous npm line* | *0.80.0 — Loom “Native Ship”* |

## License

Apache-2.0 — see [LICENSE](LICENSE).
