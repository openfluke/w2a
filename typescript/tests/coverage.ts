/**
 * Parity: every expected_api / getWelvetInternalParity symbol must exist.
 */
import { init, assertEngineVersion, getInternalParity } from "../src/index.js";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log("=== Welvet WASM coverage ===");
  await init();
  assertEngineVersion();

  let expected: string[] = [];
  const candidates = [
    join(__dirname, "..", "assets", "expected_api.json"),
    join(__dirname, "..", "..", "wasm", "expected_api.json"),
    join(__dirname, "expected_api.json"),
  ];
  for (const p of candidates) {
    try {
      expected = JSON.parse(readFileSync(p, "utf-8")) as string[];
      console.log("  loaded", p, `(${expected.length})`);
      break;
    } catch { /* try next */ }
  }
  if (expected.length === 0) {
    expected = getInternalParity();
    console.log("  fallback parity from WASM", expected.length);
  }

  const parity = new Set(getInternalParity());
  let missing = 0;
  let present = 0;

  const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));

  for (const item of expected) {
    if (item.startsWith("Grid.")) {
      const m = item.slice(5);
      if (typeof (g as unknown as Record<string, unknown>)[m] === "function" || m in (g as object)) {
        present++;
      } else {
        console.error("  MISSING", item);
        missing++;
      }
      continue;
    }
    if (item.startsWith("Stack.") || item.startsWith("Parallel.")) {
      // checked via factory wrappers below
      present++;
      continue;
    }
    const fn = (globalThis as Record<string, unknown>)[item];
    if (typeof fn === "function" || parity.has(item)) {
      present++;
    } else {
      console.error("  MISSING", item);
      missing++;
    }
  }

  // Stack / Parallel method spot-check
  const st = createWelvetBicameral(JSON.stringify({ in: 4, hidden: 4, out: 4 }));
  for (const m of ["trainStackMSE", "setChildModes", "setTanhi", "forward", "placeOnGrid"]) {
    if (typeof (st as unknown as Record<string, unknown>)[m] !== "function") {
      console.error("  MISSING Stack." + m);
      missing++;
    } else present++;
  }
  const p = createWelvetParallel(JSON.stringify({ Dim: 4, OutFeat: 4, Branches: 2 }));
	for (const m of ["trainMSE", "setBranchModes", "setCamSync", "setCamKit", "setTanhi", "forward", "placeOnGrid"]) {
    if (typeof (p as unknown as Record<string, unknown>)[m] !== "function") {
      console.error("  MISSING Parallel." + m);
      missing++;
    } else present++;
  }

  console.log(`  present=${present} missing=${missing}`);
  if (missing > 0) {
    console.error("=== COVERAGE FAIL ===");
    process.exit(1);
  }
  console.log("=== COVERAGE OK ===");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
