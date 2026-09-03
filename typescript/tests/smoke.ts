/**
 * Core smoke: version, dense place, forward, trainSGD, entity roundtrip.
 */
import { init, createGrid, assertEngineVersion, WELVET_ENGINE_VERSION } from "../src/index.js";

function fail(msg: string): never {
  console.error("FAIL:", msg);
  process.exit(1);
}

function ok(msg: string) {
  console.log("  PASS", msg);
}

async function main() {
  console.log("=== Welvet WASM smoke (", WELVET_ENGINE_VERSION, ") ===");
  await init();
  assertEngineVersion();
  ok(`engine version ${welvetEngineVersion()}`);

  const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
  if (!g || typeof g._id !== "number") fail("createGrid");
  ok(`createGrid id=${g._id}`);

  const placed = g.placeDense(JSON.stringify({ z: 0, y: 0, x: 0, l: 0, in: 4, out: 4, act: "relu", dtype: 1 }));
  if ((placed as { error?: string }).error) fail(`placeDense: ${(placed as { error: string }).error}`);
  ok("placeDense");

  const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const fwd = g.forward(input);
  if ((fwd as { error?: string }).error) fail(`forward: ${(fwd as { error: string }).error}`);
  if (!fwd.output || fwd.output.length !== 4) fail(`forward len=${fwd.output?.length}`);
  ok(`forward out=[${Array.from(fwd.output).map((x) => x.toFixed(3)).join(",")}]`);

  const target = new Float32Array([0.0, 1.0, 0.0, 0.0]);
  const tr = g.trainSGD(input, target, 0.05);
  if ((tr as { error?: string }).error) fail(`trainSGD: ${(tr as { error: string }).error}`);
  if (typeof tr.loss !== "number" || Number.isNaN(tr.loss)) fail(`bad loss ${tr.loss}`);
  ok(`trainSGD loss=${tr.loss.toFixed(6)}`);

  const ent = g.serializeEntity();
  if (!ent || (ent as unknown as { error?: string }).error) fail("serializeEntity");
  if (!(ent instanceof Uint8Array) || ent.length < 8) fail(`entity bytes=${(ent as Uint8Array)?.length}`);
  ok(`serializeEntity bytes=${ent.length}`);

  const round = DeserializeGrid(ent);
  if (!round || (round as unknown as { error?: string }).error) fail("DeserializeGrid");
  ok(`DeserializeGrid id=${round._id}`);

  const dna = g.extractDNA();
  if (!dna || dna.includes('"error"')) fail("extractDNA");
  ok("extractDNA");

  g.free();
  console.log("=== SMOKE OK ===");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
