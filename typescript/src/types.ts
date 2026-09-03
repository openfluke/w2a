/** Welvet engine version — must match Go welvetEngineVersion in apps/w2a/wasm. */
export const WELVET_ENGINE_VERSION = "1.1.1";

/** @deprecated Loom-compat alias */
export const LOOM_ENGINE_VERSION = WELVET_ENGINE_VERSION;

export const DType = {
  FLOAT64: 0,
  FLOAT32: 1,
  FLOAT16: 2,
  BFLOAT16: 3,
  FP8_E4M3: 4,
  FP8_E5M2: 5,
  INT64: 6,
  INT32: 7,
  INT16: 8,
  INT8: 9,
  UINT64: 10,
  UINT32: 11,
  UINT16: 12,
  UINT8: 13,
  INT4: 14,
  UINT4: 15,
  FP4: 16,
  INT2: 17,
  UINT2: 18,
  TERNARY: 19,
  BINARY: 20,
} as const;
export type DTypeValue = (typeof DType)[keyof typeof DType];

export const Format = {
  None: 0,
  Q8_0: 1,
  Q4_0: 2,
  Q4_1: 3,
  Q5_0: 4,
  Q5_1: 5,
} as const;

export const Backend = {
  CPUTiled: "cpu_tiled",
  SIMD: "simd",
  WebGPU: "webgpu",
} as const;

export const LayerType = {
  Dense: "Dense",
  MultiHeadAttention: "MultiHeadAttention",
  SwiGLU: "SwiGLU",
  RMSNorm: "RMSNorm",
  LayerNorm: "LayerNorm",
  Embedding: "Embedding",
  Softmax: "Softmax",
  Sequential: "Sequential",
  Residual: "Residual",
  CNN1: "CNN1",
  CNN2: "CNN2",
  CNN3: "CNN3",
  RNN: "RNN",
  LSTM: "LSTM",
  ConvT1: "ConvT1",
  ConvT2: "ConvT2",
  ConvT3: "ConvT3",
  Parallel: "Parallel",
  Stack: "Stack",
  KMeans: "KMeans",
  Mamba: "Mamba",
  Metacognition: "Metacognition",
  GDN: "GDN",
} as const;

export const TrainMode = {
  NormalBP: "NormalBP",
  StepBP: "StepBP",
  Tween: "Tween",
  TweenChain: "TweenChain",
  StepTween: "StepTween",
  StepTweenChain: "StepTweenChain",
  MeshBP: "MeshBP",
  MeshTween: "MeshTween",
  MeshTweenChain: "MeshTweenChain",
  Freeze: "Freeze",
  Shadow: "Shadow",
  Adversarial: "Adversarial",
  Memory: "Memory",
} as const;

export const CombineMode = {
  Add: "Add",
  Concat: "Concat",
  Mean: "Mean",
} as const;

export const Activation = {
  RELU: 0,
  SILU: 1,
  GELU: 2,
  TANH: 3,
  SIGMOID: 4,
  LINEAR: 5,
} as const;

export interface GridConfig {
  depth?: number;
  rows?: number;
  cols?: number;
  layers_per_cell?: number;
  backend?: string;
}

export interface PlaceSpec {
  z?: number;
  y?: number;
  x?: number;
  l?: number;
  in?: number;
  out?: number;
  dim?: number;
  act?: string;
  dtype?: number;
  format?: number;
  [key: string]: unknown;
}

export interface TrainResult {
  loss: number;
  output?: Float32Array;
  error?: string;
  mode?: string;
}

export interface ForwardResult {
  output: Float32Array;
  shape?: string;
  steps?: number;
  error?: string;
}

export interface Grid {
  _id: number;
  depth: number;
  rows: number;
  cols: number;
  layersPerCell: number;
  stackLayerCount: number;
  getInfo(): string;
  setRemoteLink(z: number, y: number, x: number, l: number, tz: number, ty: number, tx: number, tl: number): { status?: string; error?: string };
  clearRemoteLink(z: number, y: number, x: number, l: number): { status?: string; error?: string };
  convertDense(dtype: number, format: number, z?: number, y?: number, x?: number, l?: number): { status?: string; error?: string };
  applySGDDense(dW: Float64Array | number[], lr?: number, z?: number, y?: number, x?: number, l?: number): { status?: string; error?: string };
  configureTanhi(cfgJSON?: string): { status?: string; error?: string };
  placeDense(spec: string | PlaceSpec): { status?: string; error?: string };
  placeMHA(spec: string | PlaceSpec): { status?: string; error?: string };
  placeSwiGLU(spec: string | PlaceSpec): { status?: string; error?: string };
  placeRMSNorm(spec: string | PlaceSpec): { status?: string; error?: string };
  placeLayerNorm(spec: string | PlaceSpec): { status?: string; error?: string };
  placeEmbedding(spec: string | PlaceSpec): { status?: string; error?: string };
  placeSoftmax(spec: string | PlaceSpec): { status?: string; error?: string };
  placeSequential(spec: string | PlaceSpec): { status?: string; error?: string };
  placeResidual(spec: string | PlaceSpec): { status?: string; error?: string };
  placeCNN1(spec: string | PlaceSpec): { status?: string; error?: string };
  placeCNN2(spec: string | PlaceSpec): { status?: string; error?: string };
  placeCNN3(spec: string | PlaceSpec): { status?: string; error?: string };
  placeRNN(spec: string | PlaceSpec): { status?: string; error?: string };
  placeLSTM(spec: string | PlaceSpec): { status?: string; error?: string };
  placeConvT1(spec: string | PlaceSpec): { status?: string; error?: string };
  placeConvT2(spec: string | PlaceSpec): { status?: string; error?: string };
  placeConvT3(spec: string | PlaceSpec): { status?: string; error?: string };
  placeParallel(spec: string | PlaceSpec): { status?: string; error?: string };
  placeStack(spec: string | PlaceSpec): { status?: string; error?: string };
  placeKMeans(spec: string | PlaceSpec): { status?: string; error?: string };
  placeMamba(spec: string | PlaceSpec): { status?: string; error?: string };
  placeMetacognition(spec: string | PlaceSpec): { status?: string; error?: string };
  placeGDN(spec: string | PlaceSpec): { status?: string; error?: string };
  forward(data: Float32Array | number[], shape?: string | number[]): ForwardResult;
  backward(grad: Float32Array | number[], shape?: string | number[]): { gradIn?: Float32Array; error?: string };
  trainSGD(input: Float32Array | number[], targetOrShape: Float32Array | number[] | string, lrOrTarget?: number | Float32Array | number[], lr?: number): TrainResult;
  trainTween(input: Float32Array | number[], target: Float32Array | number[], lr?: number): TrainResult;
  trainMesh(input: Float32Array | number[], target: Float32Array | number[], ticks?: number, lr?: number): TrainResult;
  extractDNA(): string;
  extractBlueprint(): string;
  serializeEntity(): Uint8Array;
  setMultiCore?(multi: boolean): { status?: string; error?: string };
  getDenseWeights(z?: number, y?: number, x?: number, l?: number): Float32Array | { error: string };
  setDenseWeights(data: Float32Array, z?: number, y?: number, x?: number, l?: number): { status?: string; error?: string };
  free(): { status?: string };
}

export interface Stack {
  _id: number;
  kind: "stack";
  children: number;
  setChildModes(...modes: string[]): { status?: string; error?: string };
  setTanhi(cfgJSON: string): { status?: string; error?: string };
  trainStackMSE(input: Float32Array, target: Float32Array, mode?: string, lr?: number): TrainResult;
  trainStackCE(input: Float32Array, target: Float32Array, mode?: string, lr?: number): TrainResult;
  forward(data: Float32Array | number[]): ForwardResult;
  placeOnGrid(gridId: number, z?: number, y?: number, x?: number, l?: number): { status?: string; error?: string };
  free(): { status?: string };
}

export interface Parallel {
  _id: number;
  kind: "parallel";
  branches: number;
  setBranchModes(...modes: string[]): { status?: string; error?: string };
  setTanhi(cfgJSON: string): { status?: string; error?: string };
  setCamSync(cfgJSON: string): { status?: string; error?: string };
  setCamKit(cfgJSON: string): { status?: string; error?: string };
  trainMSE(input: Float32Array, target: Float32Array, mode?: string, lr?: number): TrainResult;
  forward(data: Float32Array | number[]): ForwardResult;
  placeOnGrid(gridId: number): { status?: string; error?: string };
  free(): { status?: string };
}

declare global {
  function welvetEngineVersion(): string;
  function loomEngineVersion(): string;
  function createWelvetGrid(json: string): Grid;
  function createLoomNetwork(json: string): Grid;
  function Forward(gridId: number, data: Float32Array, shape?: string): ForwardResult;
  function TrainStep(gridId: number, input: Float32Array, target: Float32Array, lr?: number): TrainResult;
  function TrainStepTween(gridId: number, input: Float32Array, target: Float32Array, lr?: number): TrainResult;
  function TrainStepMesh(gridId: number, input: Float32Array, target: Float32Array, ticks?: number, lr?: number): TrainResult;
  function createWelvetStepState(gridId: number): { _id: number; setInput(d: Float32Array): unknown; step(capture?: boolean): unknown; free(): unknown };
  function createWelvetTweenState(gridId: number): { _id: number; free(): unknown };
  function createWelvetBicameral(cfg: string | number): Stack;
  function createWelvetHemispheres(cfg: string): Parallel;
  function createWelvetParallel(cfg: string): Parallel;
  function listWelvetTrainModes(): string;
  function listWelvetNamedTrainModes(): string;
  function listWelvetConcreteTrainModes(): string;
  function listWelvetCreditTrainModes(): string;
  function ParseTrainMode(name: string): string | { error: string };
  function listWelvetLayerTypes(): string;
  function listWelvetDTypes(): string;
  function listWelvetFormats(): string;
  function listWelvetBackends(): string;
  function listWelvetSuiteCatalog(): string;
  function listWelvetTemplates(): string;
  function welvetPermutationOK(kind: string, dtype: number, format: number, backend: number): boolean;
  function ConfigureTanhi(gridId: number, cfgJSON?: string): { status?: string; error?: string };
  function EmitSweep(label?: string, cfgJSON?: string): { status?: string };
  function DefaultTanhiUDPPort(): number;
  function ExtractDNA(gridId: number): string;
  function CompareDNA(dnaJSON_A: string, dnaJSON_B: string): string;
  function ExtractNetworkBlueprint(gridId: number, modelID?: string): string;
  function CloneGrid(gridId: number): Grid;
  function defaultSpliceConfig(): string;
  function defaultNEATConfig(dModel: number): string;
  function SpliceDNA(gridIdA: number, gridIdB: number, cfgJSON?: string): Grid;
  function NEATMutate(gridId: number, cfgJSON?: string): Grid;
  function createWelvetNEATPopulation(
    gridId: number,
    size: number,
    cfgJSON?: string,
  ): {
    size: number;
    evolveWithFitnesses(fits: Float64Array): { status?: string; error?: string };
    bestFitness(): number;
    summary(generation?: number): string;
    error?: string;
  };
  function listWelvetActivations(): string;
  function listWelvetNativeOnlyCases(): string;
  function SerializeGrid(gridId: number): Uint8Array;
  function DeserializeGrid(data: Uint8Array): Grid;
  function SerializeEntity(gridId: number): Uint8Array | { error: string };
  function DeserializeEntity(data: Uint8Array): Grid;
  function BuildSpec(gridId: number): string;
  function PackableFormats(): string;
  function ArgMax(logits: Float32Array): number;
  function SampleTopK(logits: Float32Array, k: number, temperature: number): number;
  function NewTokenizerFromJSON(json: string | Uint8Array): { encode(text: string, addSpecial?: boolean): Uint32Array; decode(ids: Uint32Array, skipSpecial?: boolean): string };
  function BuildTransformerPrompt(user: string, system?: string): string;
  function LucyAvailability(inferMs: number, trainMs: number): number;
  function LucyScore(throughput: number, availability: number, acc: number): number;
  function LucySoftAccBatch(pred: Float32Array, target: Float32Array): number;
  function welvetGC(): { status?: string };
  function getWelvetInternalParity(): string;
  // seed / fountain / memory / helpers / weights
  function SeedFrom(partsJSON: string): string | { error: string };
  function InitGrid(gridId: number, seed: string | number): { status?: string; error?: string };
  function GridFingerprint(gridId: number): string | { error: string };
  function BuildDenseManifest(topologySeed: string | number, sizesJSON: string, dtypesJSON?: string): string | { error: string };
  function BuildDenseGridFromManifest(manifestJSON: string): Grid;
  function InitStoreHe(storeId: number, inputSize: number, seed: string | number): { status?: string; error?: string };
  function FountainRecoverWeightBlobs(blobs: Uint8Array[], seed?: string | number, loss?: number, maxOverhead?: number): { recovered: Uint8Array[]; received: number; sprayed: number; error?: string };
  function FountainLTRoundTrip(sources: Uint8Array[], seed?: string | number): { ok: boolean; received: number; sprayed: number; k: number; error?: string };
  function PackGridWeights(gridId: number): Uint8Array | { error: string };
  function UnpackGridWeights(gridId: number, blob: Uint8Array): { status?: string; error?: string };
  function MemoryFromGrid(gridId: number): string;
  function ReleaseTransient(): { status?: string };
  function SetMemoryHistoryRecording(enabled: boolean): { status?: string };
  function GraftGrids(gridIds: number[] | string, combine?: string): Parallel;
  function GraftToGrid(gridIds: number[] | string): Grid;
  function ResidualGraftGrid(gridId: number): Grid;
  function TemplateBuildPrompt(template: string, user: string, system?: string): string;
  function EnsembleMajorityVote(votesJSON: string): string;
  function EvaluatePrediction(sampleIndex: number, expected: number, actual: number): string;
  function IntrospectGrid(gridId: number): string;
  function ProbeDeepGeometry(geomsJSON: string): string;
  function createWelvetStore(rows: number, cols: number, dtype?: number, format?: number, data?: Float32Array): Store;
  function listWelvetCameralPolyKinds(): string;
  function TrainCameralPoly(kind: string, mode: string, dtype?: number, format?: number): { loss: number; note: string; mode: string; kind: string; error?: string };
}

export interface Store {
  _id: number;
  rows: number;
  cols: number;
  dtype: number;
  format: number;
  applySGD(dW: Float64Array | number[], lr?: number): { status?: string; error?: string };
  setDType(dtype: number): { status?: string; error?: string };
  pack(format: number): { status?: string; error?: string };
  convert(dtype: number, format: number): { status?: string; error?: string };
  flattenF32(): Float32Array | { error: string };
  retainsF32Master(): boolean;
  f32BufferLen(): number;
  free(): { status?: string };
}
