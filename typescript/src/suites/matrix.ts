/** Shared suite matrix for Bun + browser runners. */
export const LAYER_PLACE_SPECS = [
  { method: "placeDense", spec: { in: 8, out: 8, act: "relu", dtype: 1 } },
  { method: "placeSwiGLU", spec: { dim: 8, InputDim: 8, IntermediateDim: 16 } },
  { method: "placeRMSNorm", spec: { dim: 8 } },
  { method: "placeParallel", spec: { dim: 8, OutFeat: 8, Branches: 2 } },
  { method: "placeStack", spec: { dim: 8 } },
] as const;

export async function runLayerPlaceMatrix(createGrid: () => any): Promise<{ pass: number; fail: number }> {
  let pass = 0, fail = 0;
  for (const { method, spec } of LAYER_PLACE_SPECS) {
    const g = createGrid();
    const r = g[method](JSON.stringify({ z: 0, y: 0, x: 0, l: 0, ...spec }));
    if (r?.error) fail++; else pass++;
    g.free?.();
  }
  return { pass, fail };
}
