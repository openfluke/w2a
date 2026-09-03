#!/usr/bin/env bun
import {
  init,
  assertEngineVersion,
  createBicameral,
  createHemispheres,
  listNamedTrainModes,
} from "@openfluke/welvet";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

console.log("=== tdeploy/bun cameral ===");
await init();
assertEngineVersion();

const modes = listNamedTrainModes();
assert(modes.length >= 10, `modes=${modes.length}`);
const x = new Float32Array(4).fill(0.1);
const y = new Float32Array([1, 0, 0, 0]);

const stack = createBicameral({ in: 4, hidden: 4, out: 4 });
const r1 = stack.trainStackMSE(x, y, modes[0], 0.05);
assert(!r1.error, `bicameral: ${r1.error}`);
console.log(`  PASS bicameral ${modes[0]} loss=${r1.loss}`);

const hem = createHemispheres({ dim: 4, n: 2, combine: "add" });
hem.setBranchModes(JSON.stringify([modes[0], modes[0]]));
hem.setCamSync(JSON.stringify({ Enabled: true, Alpha: 1 }));
const r2 = hem.trainMSE(x, y, modes[0], 0.05);
assert(!r2.error, `hemispheres: ${r2.error}`);
console.log(`  PASS hemispheres+CamSync loss=${r2.loss}`);

stack.free?.();
hem.free?.();
console.log("=== tdeploy/bun cameral OK ===");
