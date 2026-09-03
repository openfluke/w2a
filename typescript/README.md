# @openfluke/welvet

**Welvet 1.1.1** — Isomorphic TypeScript/WASM bindings for the [Welvet](https://github.com/openfluke/welvet) engine: layers, dtypes, cameral TrainModes, CamSync/CamKit, Tanhi, DNA/NEAT, Lucy.

[![npm version](https://img.shields.io/npm/v/@openfluke/welvet.svg)](https://www.npmjs.com/package/@openfluke/welvet)

## Breaking change from 0.80 (Loom)

`@openfluke/welvet@0.80.0` was Loom-era. **1.1.1** targets Welvet:

| 0.80 (Loom) | 1.1.1 (Welvet) |
|-------------|----------------|
| `LOOM_ENGINE_VERSION` | `WELVET_ENGINE_VERSION` (`"1.1.1"`) |
| `createLoomNetwork` | `createWelvetGrid` / `createGrid` |
| — | Cameral: bicameral, TrainModes, CamSync, CamKit, Tanhi |

Loom aliases (`createLoomNetwork`, `loomEngineVersion`) still work for migration.

## Install

```bash
npm install @openfluke/welvet@1.1.1
# or
bun add @openfluke/welvet@1.1.1
```

## Quick start

```ts
import { init, createGrid, assertEngineVersion } from "@openfluke/welvet";

await init();
assertEngineVersion();

const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
g.placeDense(JSON.stringify({ in: 4, out: 4, act: "relu", dtype: 1 }));
const out = g.forward(new Float32Array([0.1, 0.2, 0.3, 0.4]));
const step = g.trainSGD(new Float32Array(4).fill(0.1), new Float32Array([0, 1, 0, 0]), 0.05);
console.log(step.loss, out.output);
```

Cameral:

```ts
import { createBicameral, listNamedTrainModes } from "@openfluke/welvet";

const modes = listNamedTrainModes();
const stack = createBicameral({ in: 4, hidden: 4, out: 4 });
stack.trainStackMSE(input, target, modes[0], 0.05);
```

## Test matrix (no publish required)

| Profile | Cells (order) | Command |
|---------|---------------|---------|
| quick | ~100 | `npm run test:all:quick` |
| mid | ~900 | `npm run test:all:mid` |
| **mega** | **~200k+** (Go w2a-scale product) | `npm run test:all` |

Mega axes (same idea as native w2a gap census):

- **layers × 34 dtypes × 21 formats × 3 backends × {place,forward,train,entity}**
- **cameral** arches × all named TrainModes × (dtypes@FormatNone ‖ quants@f32) × backends
- **volumetric** dense 1³/2³/3³ × dtypes/formats × backends
- **weights** ApplySGD × all dtypes · **seed/fountain/helpers/memory** portable Cases
- **serialization** convert dtype/quant · **train_modes** × dtypes + concrete×formats
- **train_modes × layers** — 21 cameral-poly kinds × 31 named TrainModes × (34 dtypes + quants@f32)

Native-only (explicit SKIP): donate TCP, hardware audit, host SIMD Plan-9, WebGPU↔CPU parity, FS Load/Save paths.

```bash
cd apps/w2a/typescript
npm run build:all
npm run test:all:quick   # smoke accessibility
npm run test:all         # MEGA — quiet progress every 2k cells; FAIL fails the process
npm run serve            # http://localhost:3000/matrix.html
```

## Build from source (`apps/w2a`)

WASM uses `syscall/js` over Welvet (not CGO/`cabi`). Broken non-amd64 SIMD stubs are patched at **build time** via Go `-overlay` under `wasm/overlays/` — the Welvet tree is not modified.

```bash
cd apps/w2a/wasm && bash build.sh
cd ../typescript && npm install && npm run build && npm test
```

## License

Apache-2.0
