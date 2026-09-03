/** Catalog mirroring apps/w2a/suites + portable WASM ports. */
export const W2A_SUITE_CATALOG = [
  // Layer suites (Go suites/*)
  "dense", "mha", "swiglu", "rmsnorm", "layernorm", "embedding", "softmax",
  "sequential", "residual", "cnn1", "cnn2", "cnn3", "rnn", "lstm",
  "convt1", "convt2", "convt3", "parallel", "kmeans", "mamba", "metacognition", "gdn",
  // Systems / runtime
  "dna", "evolution", "tween", "step", "serialization", "seven", "weights",
  "tanhi", "telemetry", "lucy", "sampling", "entity",
  // Honesty / portable extras (WASM ports of Go Cases)
  "honesty", "acts", "native_only", "step_tween",
  // Cameral / train
  "train_modes", "cameral",
  // Portable stubs now wrapped in WASM
  "seed", "fountain", "helpers", "memory",
  "train_modes_layers",
] as const;

export type SuiteId = (typeof W2A_SUITE_CATALOG)[number];

/** Place method + default geometry per layer (WASM place* JSON). */
export interface LayerPlaceDef {
  id: string;
  method: string;
  /** Base place spec (z/y/x/l filled by runner). */
  spec: Record<string, unknown>;
  /** Input length for forward/train (flattened). */
  inLen: number;
  /** Target length for trainSGD (fallback; mega prefers forward.output.length). */
  outLen: number;
  /** Optional shape JSON for forward. */
  shape?: number[];
  /** Skip trainSGD only for known weightless / non-SGD layers. */
  skipTrain?: boolean;
}

export const LAYER_DEFS: LayerPlaceDef[] = [
  { id: "dense", method: "placeDense", spec: { in: 8, out: 8, act: "relu" }, inLen: 8, outLen: 8 },
  { id: "mha", method: "placeMHA", spec: { dim: 8, DModel: 8, NumHeads: 2, SeqLen: 4 }, inLen: 32, outLen: 32, shape: [1, 4, 8] },
  { id: "swiglu", method: "placeSwiGLU", spec: { dim: 8, InputDim: 8, IntermediateDim: 16 }, inLen: 8, outLen: 8 },
  { id: "rmsnorm", method: "placeRMSNorm", spec: { dim: 8 }, inLen: 8, outLen: 8 },
  { id: "layernorm", method: "placeLayerNorm", spec: { dim: 8 }, inLen: 8, outLen: 8 },
  { id: "embedding", method: "placeEmbedding", spec: { VocabSize: 32, EmbeddingDim: 8, SeqLen: 4 }, inLen: 4, outLen: 32, shape: [1, 4] },
  { id: "softmax", method: "placeSoftmax", spec: { dim: 8 }, inLen: 8, outLen: 8, skipTrain: true },
  { id: "sequential", method: "placeSequential", spec: { dim: 8, Depth: 2 }, inLen: 8, outLen: 8 },
  { id: "residual", method: "placeResidual", spec: { dim: 8, Depth: 1 }, inLen: 8, outLen: 8 },
  { id: "cnn1", method: "placeCNN1", spec: { InChannels: 1, Filters: 4, SeqLen: 16, Kernel: 3 }, inLen: 16, outLen: 56, shape: [1, 1, 16] },
  { id: "cnn2", method: "placeCNN2", spec: { InChannels: 1, Filters: 4, Height: 8, Width: 8, Kernel: 3 }, inLen: 64, outLen: 144, shape: [1, 1, 8, 8] },
  { id: "cnn3", method: "placeCNN3", spec: { InChannels: 1, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 3 }, inLen: 64, outLen: 16, shape: [1, 1, 4, 4, 4] },
  { id: "rnn", method: "placeRNN", spec: { InputSize: 8, HiddenSize: 8, SeqLen: 4 }, inLen: 32, outLen: 32, shape: [1, 4, 8] },
  { id: "lstm", method: "placeLSTM", spec: { InputSize: 8, HiddenSize: 8, SeqLen: 4 }, inLen: 32, outLen: 32, shape: [1, 4, 8] },
  { id: "convt1", method: "placeConvT1", spec: { InChannels: 4, Filters: 2, SeqLen: 8, Kernel: 3 }, inLen: 32, outLen: 20, shape: [1, 4, 8] },
  { id: "convt2", method: "placeConvT2", spec: { InChannels: 4, Filters: 2, Height: 4, Width: 4, Kernel: 3 }, inLen: 64, outLen: 72, shape: [1, 4, 4, 4] },
  { id: "convt3", method: "placeConvT3", spec: { InChannels: 2, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 3 }, inLen: 128, outLen: 250, shape: [1, 2, 4, 4, 4] },
  { id: "parallel", method: "placeParallel", spec: { dim: 8, OutFeat: 8, Branches: 2 }, inLen: 8, outLen: 8 },
  { id: "stack", method: "placeStack", spec: { dim: 8, act: "relu" }, inLen: 8, outLen: 8 },
  { id: "kmeans", method: "placeKMeans", spec: { FeatureDim: 8, NumClusters: 4 }, inLen: 8, outLen: 4 },
  { id: "mamba", method: "placeMamba", spec: { DModel: 8, DState: 8, SeqLen: 4 }, inLen: 32, outLen: 32, shape: [1, 4, 8] },
  { id: "metacognition", method: "placeMetacognition", spec: { Dim: 8 }, inLen: 8, outLen: 8 },
  { id: "gdn", method: "placeGDN", spec: { HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 3 }, inLen: 8, outLen: 8, shape: [1, 1, 8] },
];
