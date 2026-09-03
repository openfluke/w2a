/**
 * Systems + model + Lucy smoke.
 */
import { init, createGrid, assertEngineVersion } from "../src/index.js";

async function main() {
  console.log("=== Welvet WASM systems/model/lucy ===");
  await init();
  assertEngineVersion();

  const g = createGrid();
  g.placeDense(JSON.stringify({ in: 4, out: 4, act: "relu", dtype: 1 }));

  const dna = ExtractDNA(g._id);
  if (!dna || dna.includes('"error"')) throw new Error("ExtractDNA");
  console.log("  PASS ExtractDNA");

  const bp = ExtractNetworkBlueprint(g._id, "wasm-test");
  if (!bp || bp.includes('"error"')) throw new Error("blueprint");
  console.log("  PASS ExtractNetworkBlueprint");

  const clone = CloneGrid(g._id);
  if (!clone || (clone as unknown as { error?: string }).error) throw new Error("CloneGrid");
  console.log("  PASS CloneGrid", clone._id);

  const ser = SerializeGrid(g._id);
  if (!(ser instanceof Uint8Array) || ser.length < 4) throw new Error("SerializeGrid");
  const back = DeserializeGrid(ser);
  if (!back || (back as unknown as { error?: string }).error) throw new Error("DeserializeGrid");
  console.log("  PASS Serialize/DeserializeGrid");

  const logits = new Float32Array([0.1, 0.9, 0.2, 0.05]);
  if (ArgMax(logits) !== 1) throw new Error("ArgMax");
  const tok = SampleTopK(logits, 2, 1.0);
  if (typeof tok !== "number") throw new Error("SampleTopK");
  console.log("  PASS ArgMax/SampleTopK", tok);

  const avail = LucyAvailability(1.0, 2.0);
  const score = LucyScore(100, avail, 0.9);
  const soft = LucySoftAccBatch(new Float32Array([0.8, 0.2]), new Float32Array([1, 0]));
  if (typeof avail !== "number" || typeof score !== "number" || typeof soft !== "number") {
    throw new Error("Lucy APIs");
  }
  console.log("  PASS Lucy", { avail, score, soft });

  const step = createWelvetStepState(g._id);
  step.setInput(new Float32Array([0.1, 0.2, 0.3, 0.4]));
  const s = step.step(false);
  if ((s as { error?: string }).error) throw new Error("step: " + (s as { error: string }).error);
  console.log("  PASS StepState");

	const tw = createWelvetTweenState(g._id);
  tw.free();
  console.log("  PASS TweenState");

  const spliceCfg = defaultSpliceConfig();
  if (!spliceCfg) throw new Error("defaultSpliceConfig");
  const neatCfg = defaultNEATConfig(8);
  if (!neatCfg) throw new Error("defaultNEATConfig");
  console.log("  PASS evolution configs");

  const prompt = BuildTransformerPrompt("hi", "sys");
  if (typeof prompt !== "string" || !prompt.includes("hi")) throw new Error("BuildTransformerPrompt");
  console.log("  PASS BuildTransformerPrompt");

  welvetGC();
  console.log("=== SYSTEMS OK ===");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
