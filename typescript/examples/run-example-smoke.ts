/**
 * Mirror of example book + cam WASM smokes (also under /home/openfluke/git/example).
 * Run: npx tsx examples/run-example-smoke.ts
 */
import { init, createGrid, createBicameral, createHemispheres, listNamedTrainModes, assertEngineVersion } from "../src/index.js";

await init();
assertEngineVersion();

const g = createGrid();
g.placeDense(JSON.stringify({ in: 8, out: 4, act: "relu", dtype: 1 }));
const x = Float32Array.from({ length: 8 }, (_, i) => i * 0.1);
console.log("dense", g.forward(x).output.length);

const modes = listNamedTrainModes();
const st = createBicameral({ in: 4, hidden: 4, out: 4 });
console.log("bicameral", st.trainStackMSE(new Float32Array(4).fill(0.1), new Float32Array([1, 0, 0, 0]), modes[0], 0.05).loss);

const hem = createHemispheres({ dim: 4, n: 2, combine: "add" });
hem.setCamSync(JSON.stringify({ Enabled: true, Alpha: 1 }));
console.log("hemispheres", hem.trainMSE(new Float32Array(4).fill(0.1), new Float32Array([1, 0, 0, 0]), modes[0], 0.05).loss);

console.log("example smoke OK");
