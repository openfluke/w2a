/**
 * W2A WASM matrix.
 *   --quick  ~100 cells
 *   --full   ~900 cells (default)
 *   --mega   ~200k+ cells (layer×dtype×format×backend×ops + cameral + volumetric)
 */
import { init, assertEngineVersion } from "../src/index.js";
import { runAllW2ASuites } from "../src/suites/index.js";

const profile = process.argv.includes("--mega")
  ? "mega"
  : process.argv.includes("--quick")
    ? "quick"
    : "full";

await init();
assertEngineVersion();
const t0 = performance.now();
await runAllW2ASuites({ profile, failOnFail: true });
console.log(`=== test_all OK (${profile}) in ${((performance.now() - t0) / 1000).toFixed(1)}s ===`);
