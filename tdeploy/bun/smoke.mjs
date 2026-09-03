#!/usr/bin/env bun
/**
 * Bun backend — registry @openfluke/welvet@1.1.1
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  init,
  createGrid,
  assertEngineVersion,
  WELVET_ENGINE_VERSION,
  engineVersion,
  DType,
  seedFrom,
} from "@openfluke/welvet";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgPath = Bun.resolveSync("@openfluke/welvet/package.json", here);
const pkg = await Bun.file(pkgPath).json();
const root = path.dirname(pkgPath);

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

console.log("=== tdeploy/bun smoke ===");
console.log(`  runtime: bun ${Bun.version}`);
console.log(`  resolved: ${root}`);
assert(path.resolve(root).startsWith(path.resolve(here, "node_modules")), `not under bun/node_modules: ${root}`);
assert(pkg.version === "1.1.1", `pkg ${pkg.version}`);
assert(WELVET_ENGINE_VERSION === "1.1.1", `const ${WELVET_ENGINE_VERSION}`);

await init();
assertEngineVersion();
assert(engineVersion() === "1.1.1", `engine ${engineVersion()}`);
console.log(`  PASS engine ${engineVersion()}`);

const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
const placed = g.placeDense(
  JSON.stringify({ in: 8, out: 4, act: "relu", dtype: DType.FLOAT32 }),
);
assert(!placed?.error, `placeDense: ${placed?.error}`);

const x = new Float32Array(8);
for (let i = 0; i < 8; i++) x[i] = 0.1 * (i + 1);
const fwd = g.forward(x);
assert(!fwd.error && fwd.output?.length === 4, `forward: ${fwd.error}`);
console.log(`  PASS forward len=${fwd.output.length}`);

const tr = g.trainSGD(x, new Float32Array([1, 0, 0, 0]), 0.05);
assert(!tr.error && typeof tr.loss === "number", `trainSGD: ${tr.error}`);
console.log(`  PASS trainSGD loss=${tr.loss}`);

const ent = g.serializeEntity();
assert(ent instanceof Uint8Array && ent.length > 8, "entity");
const back = globalThis.DeserializeGrid(ent);
assert(back && typeof back.forward === "function", "DeserializeGrid");
console.log(`  PASS entity ${ent.length}B`);
console.log(`  PASS seed ${seedFrom("tdeploy-bun", "1.1.1", true)}`);
back.free?.();
g.free();
console.log("=== tdeploy/bun smoke OK ===");
