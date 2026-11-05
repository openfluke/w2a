# w2a — full leaf test inventory

Generated from `suites/**` `[]Case{ Name: ... }` and `tests/**` `func Test*`.

## Summary

| Kind | Count |
|------|------:|
| Suite leaf cases (all `Name:` in `[]*Case{}`) | **6163** |
| — layer / system suites | 474 |
| — `go_v_*` native parity suites | 5689 |
| Go `func Test*` entrypoints | **470** |

A `Test*` often runs an entire `Cases()` list (many leaf cases). Prefer leaf counts for “how many checks”.

## Layer / system suite leaf cases

| Suite | Leaf cases |
|-------|----------:|
| `parallel` | 36 |
| `dense` | 25 |
| `mha` | 23 |
| `softmax` | 23 |
| `cnn1` | 18 |
| `layernorm` | 18 |
| `rmsnorm` | 18 |
| `swiglu` | 18 |
| `embedding` | 17 |
| `gdn` | 17 |
| `lstm` | 17 |
| `rnn` | 17 |
| `cnn2` | 15 |
| `cnn3` | 15 |
| `residual` | 15 |
| `sequential` | 15 |
| `convt1` | 14 |
| `convt2` | 14 |
| `convt3` | 14 |
| `kmeans` | 14 |
| `mamba` | 14 |
| `metacognition` | 14 |
| `step` | 13 |
| `helpers` | 11 |
| `tween` | 10 |
| `serialization` | 9 |
| `seven` | 9 |
| `seed` | 8 |
| `dna` | 6 |
| `evolution` | 6 |
| `memory` | 3 |
| `donate` | 2 |
| `fountain` | 2 |
| `hardware` | 2 |
| `weights` | 2 |
| **Total** | **474** |

### suites/cnn1 (18)

- Forward smoke (same-pad identity-ish)
- Output length / shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 tiled FormatNone f32 WebGPU↔CPU parity
- §12 im2col quant WebGPU smoke (Q4_0)
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/cnn2 (15)

- Forward smoke (same-pad 2D smoke)
- Output spatial / shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 tiled FormatNone f32 WebGPU↔CPU parity
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/cnn3 (15)

- Forward smoke (same-pad 3D smoke)
- Output spatial / shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 tiled FormatNone f32 WebGPU↔CPU parity
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/convt1 (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/convt2 (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/convt3 (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/dense (25)

- Forward CPU tiled (identity + bias) [float32 acts]
- Forward CPU tiled [float64 acts]
- Numeric acts — float64/int8/uint16 on CPU+SIMD (not f32-only)
- Numeric TRAIN — act Tensor[T] × matching weight dtype (native SGD, no f32 master)
- Backward CPU tiled (dW spot-check)
- Grad verify — CPU vs SIMD agreement + finite-diff dW
- SIMD Plan 9 DotTile (FP32 + ReLU)
- SIMD Plan 9 enabled on this arch
- Quant Q4_0 pack/unpack round-trip
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (native stream)
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- CPU tiled matrix — all dtype × all quant
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- SIMD fused k-cache — EnsureKSIMDCache builds Int8QS (no F32 inflate)
- SIMD fused k/IQ/Affine — CPU packed MatVec parity (no F32 inflate)
- SIMD fused AffinePacked — CPU vs SIMD fwd + MatVecPackedBlob
- §12 AffinePacked WebGPU GEMVT CPU↔GPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/dna (6)

- Extract+compare immutable smoke (Dense)
- Detect weight drift (cosine < 1 after mutate)
- Multi-layer Ops DNA (Dense/RMS/SwiGLU/MHA/…)
- MATRIX — FormatNone × all 34 dtypes × all layers
- MATRIX — all quants × Float32 × all layers
- FULL CENSUS — all layers × all dtypes × all quants

### suites/donate (2)

- Frame round-trip
- TCP PutModel + Infer stub echo

### suites/embedding (17)

- Forward smoke (Embedding gather)
- Seq layout [batch,seq]→[batch,seq,emb] smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 gather/scatter WebGPU↔CPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/evolution (6)

- Splice blend smoke (Dense)
- NEAT mutate + population one-gen smoke
- Multi-layer clone+splice (Dense/RMS/SwiGLU + quant)
- MATRIX — FormatNone × all 34 dtypes × all layers (clone+splice)
- MATRIX — all quants × Float32 × all layers (clone+splice)
- FULL CENSUS — all layers × all dtypes × all quants

### suites/fountain (2)

- LT peel recovers sources (no loss)
- RecoverWeightBlobs under 10% loss

### suites/gdn (17)

- ForwardDecode smoke
- Forward [B,T,H] smoke
- seqmix KindLinearAttn
- Backward shapes + GradWSize
- Backward matches finite-diff (Out proj, exact)
- ApplyGradSGD reduces loss
- Grad verify — CPU vs SIMD agreement
- WebGPU ForwardWebGPU/BackwardWebGPU gate on device
- CPU FormatNone × all dtypes (Float32-only; gaps recorded)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — None/BinaryPacked × CPU/SIMD/WebGPU
- SIMD FormatNone × all dtypes (Float32-only; gaps recorded)
- SIMD+WebGPU None/BinaryPacked (fwd+bwd)
- GAP CENSUS — supported GDN permutations
- TRAIN volumetric — FormatNone × dtypes × backends × 1³/2³/3³
- TRAIN volumetric — None/BinaryPacked × backends × 1³/2³/3³

### suites/hardware (2)

- Audit non-empty OS/CPU
- ToJSON parses

### suites/helpers (11)

- Grafting GraftGrids Dense branches
- Grafting GraftToGrid heterogeneous Dense+MHA
- Grafting ResidualGraftGrid Dense ok, non-Dense errors
- Templates ChatML BuildPrompt
- Ensemble MajorityVote + matches
- Clustering KMeans + grouping DetectMHA
- Observer stats + BufferObserver
- Introspection GetMethods on Grid
- Evaluation EvaluatePrediction
- Universal ProbeDeepGeometry + Mount
- Pipeline TokenTimelineSummary

### suites/kmeans (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/layernorm (18)

- Forward smoke (γ≈1 β≈0)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dγ/dβ spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 SIMD full scale CPU↔SIMD parity
- §12 WebGPU LayerNorm bwd CPU↔GPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/lstm (17)

- Forward smoke (LSTM gates)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 fused FormatNone f32 WebGPU↔CPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/mamba (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/memory (3)

- Footprint FromGrid
- History record when enabled
- ReleaseTransient

### suites/metacognition (14)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/mha (23)

- Forward causal+RoPE smoke (1 head, seq=2)
- GQA smoke (4 heads / 2 kv)
- Encoder bidirectional + DiffusionSelf presets
- Cross-attn ForwardWithContext (diffusion/encoder-decoder)
- Sliding-window causal + ALiBi causal
- Prefix-LM mask smoke
- Backward finite-diff dWq spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 SoftmaxSigmoid fwd+bwd smoke
- §12 train Dropout mask + bwd + Validate reject
- §12 GPU attn gate + CPU↔WebGPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/parallel (36)

- Forward smoke
- Shape smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³
- POLY ops — every major layer kind as Parallel branch (fwd+bwd)
- POLY matrix — Dense/MHA/SwiGLU/RMSNorm × dtypes/quants
- POLY TRAIN — all layer kinds × all 34 dtypes (weight-delta)
- NESTED — Parallel of Parallel
- NESTED — Parallel of Sequential Dense stack
- NESTED — Sequential mixed Dense/RMSNorm/SwiGLU
- NESTED — Residual mixed F + Parallel ResidualGraft
- NESTED TRAIN — depth/combine nests with weight-delta
- CAMERAL — Bicameral sandwich fwd+bwd+SGD
- CAMERAL — nested Stack-in-Parallel
- CAMERAL — Hemispheres(n=3) Sandwich
- CAMERAL TRAIN — all 9 test41 modes + mixed hemispheres + grid
- CAMERAL CREDIT — Split/FastProxy/Sparse/Alt × Bi/hemi/combine/poly
- CAMERAL CREDIT GRIDS — Split/FastProxy/Sparse/Alt/Mesh × 1³/2³/3³
- TEST49 — all train modes × 1³/2³/3³ × all layer kinds (origin smoke)
- CAMERAL POLY — all layer kinds as hemispheres × all train modes
- CAMERAL MIXED — hemi3/9 distinct per-hemi train modes (Dense smoke)
- CAMERAL MIXED — Dense hemi9 distinct modes × all 34 dtypes
- CAMERAL POLY MIXED — all kinds × hemi3/9 distinct modes (float32)
- CAMERAL POLY MIXED — all kinds × hemi9 distinct modes × all 34 dtypes
- CAMERAL PERM — FormatNone × dtypes × backends × test41+credit+mesh × arches
- CAMERAL PERM — quants × backends × test41+credit+mesh × arches (Float32)

### suites/residual (15)

- Forward smoke (Residual y=F(x)+x)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³
- MIXED — RMSNorm+SwiGLU F fwd+bwd+SGD

### suites/rmsnorm (18)

- Forward smoke (γ≈1)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dγ spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 SIMD full scale CPU↔SIMD parity
- §12 WebGPU RMSNorm bwd CPU↔GPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/rnn (17)

- Forward smoke (tanh RNN)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 fused FormatNone f32 WebGPU↔CPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/seed (8)

- SeedFrom deterministic
- He-init round-trip invert
- Dense manifest build+grid
- Infinite dense override round-trip
- Infinite MHA manifest round-trip
- Infinite SwiGLU manifest round-trip
- Infinite CNN1 manifest round-trip
- InitGrid + GridFingerprint on mixed grid (mha/swiglu/cnn1)

### suites/sequential (15)

- Forward smoke (Dense→Dense Sequential)
- Seq layout [batch,seq,dim] smoke
- Backward finite-diff dW spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³
- MIXED — Dense→RMSNorm→SwiGLU fwd+bwd+SGD

### suites/serialization (9)

- Serialize/Deserialize Dense round-trip forward
- Native encode stable
- FormatNone × all 34 dtypes bit-stable JSON
- Packable quants bit-stable JSON (Dense)
- Multi-layer Ops JSON round-trip
- ENTITY Save/Load Dense + forward
- ENTITY multi-layer bit-stable
- ALL layers × dtype/quant — create, KB sizes, train, JSON+ENTITY reload
- ALL layers × convert permutations — KB sizes + native save/reload

### suites/seven (9)

- Dense repeat-forward determinism (CPU/SIMD/WebGPU)
- Dense SC↔MC fwd+bwd × all 34 FormatNone
- Dense SC↔MC fwd+bwd × all quants (f32 pack)
- Dense multi-epoch train loss↓ + SC/MC/SIMD train parity
- Dense ENTITY save/load before+after train
- Dense S/M/L shape tiers (fwd+bwd+1-step train)
- Volumetric 7-Dense/cell × 1³/2³/3³ SC↔MC + short train
- Peak layers repeat-det + SC↔MC + short train (f32 + Q8_0)
- Seven-style summary table

### suites/softmax (23)

- Forward smoke (Softmax last-axis)
- Shape-preserving [batch,seq,dim] smoke
- Variant Gumbel forward smoke
- Variant Masked forward smoke
- Variant Sparsemax forward smoke
- Variant Entmax forward smoke
- Variant Hierarchical grid smoke
- Backward finite-diff dX spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 exotic Softmax WebGPU (Gumbel/Masked/Sparse/Entmax)
- §12 SIMD SoftmaxF32 CPU↔SIMD parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd, weightless ALU)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32, weightless)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/step (13)

- Dense StepForward smoke (1 tick → post)
- Dense StepForward+Backward history smoke
- Dense StepMeshTween gap reduce smoke
- Remote-link 2-cell mesh tick smoke
- SIMD smoke — BackendSIMD StepMesh (Dense/SwiGLU/MHA/RMS)
- TIMED — Dense FormatNone × 34 dtypes · CPU vs SIMD
- TIMED — Dense all quants × Float32 · CPU vs SIMD
- TIMED — all layers FormatNone Float32 · CPU vs SIMD
- MATRIX — FormatNone × all 34 dtypes × all layers × CPU/SIMD
- MATRIX — all quants × Float32 × all layers × CPU/SIMD
- FULL CENSUS — all layers × all dtypes × all quants × CPU/SIMD
- CROSS-NUMERIC TRAIN smoke — all kinds × sample W×A (acts not f32-only)
- CROSS-NUMERIC TRAIN full — all kinds × 34 dtypes × 15 act hosts

### suites/swiglu (18)

- Forward SiLU-gate smoke (identity-ish)
- Seq layout [batch,seq,in] smoke
- Backward finite-diff dWg spot-check
- Grad verify — CPU vs SIMD agreement
- WebGPU hard-errors without device (no host fake)
- §12 SIMD SiluMul CPU↔SIMD parity
- §12 WebGPU SwiGLU fuse bwd CPU↔GPU parity
- Repeat-forward determinism
- SC↔MC fwd+bwd determinism
- CPU tiled FormatNone × all 34 dtypes (fwd+bwd)
- ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU
- TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU
- TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)
- SIMD FormatNone × all 34 dtypes (fwd+bwd)
- SIMD+WebGPU all quant formats (fwd+bwd, Float32)
- GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)
- TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³
- TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³

### suites/tween (10)

- Dense StepTween gap reduce smoke
- Multi-layer chain-rule + layerwise (Dense+RMSNorm)
- SwiGLU layerwise gaps smoke
- SIMD smoke — BackendSIMD DotTile chain-rule (Dense/SwiGLU/MHA/RMS)
- TIMED — Dense FormatNone × 34 dtypes · CPU vs SIMD (ns + speedup)
- TIMED — Dense all quants × Float32 · CPU vs SIMD (ns + speedup)
- TIMED — all layers FormatNone Float32 · CPU vs SIMD (ns + speedup)
- MATRIX — FormatNone × all 34 dtypes × all layers × CPU/SIMD
- MATRIX — all quants × Float32 × all layers × CPU/SIMD
- FULL CENSUS — all layers × all dtypes × all quants × CPU/SIMD

### suites/weights (2)

- Native SGD — FormatNone × all 34 dtypes (no retained f32 scratch)
- Native SGD — float64 in-dtype ALU (no f32 hop)

## Native parity (`go_v_*`) leaf cases

| Backend | Leaf cases | Unique names | Case files |
|---------|----------:|-------------:|----------:|
| `go_v_cpp` | 1873 | 1637 | 68 |
| `go_v_cuda` | 939 | 863 | 47 |
| `go_v_metal` | 961 | 883 | 48 |
| `go_v_mlx` | 22 | 22 | 1 |
| `go_v_qnn` | 16 | 16 | 1 |
| `go_v_rocm` | 939 | 863 | 47 |
| `go_v_vulkan` | 939 | 863 | 47 |
| **Total** | **5689** | | |

### suites/go_v_cpp (1873 leaf cases)

#### `adversarial_train.go` (63)

- Parallel Adversarial∥NormalBP Float32 — exact bits
- Parallel Adversarial∥Adversarial Float32 — exact bits
- Parallel Adversarial∥NormalBP × all 34 dtypes — spectrum
- Parallel Adversarial volume 2×2×2×1 Float32 — exact bits (≡ FamilyBP)
- Dense Adversarial multi-hop Float32 — exact bits (W/loss)
- Dense Adversarial 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Adversarial 1-hop Float32 — exact bits
- RMSNorm Adversarial 1-hop × all 34 dtypes — spectrum
- LayerNorm Adversarial 1-hop Float32 — exact bits
- LayerNorm Adversarial 1-hop × all 34 dtypes — spectrum
- Softmax Adversarial 1-hop Float32 — exact bits (loss)
- Softmax Adversarial 1-hop × all 34 dtypes — spectrum
- SwiGLU Adversarial 1-hop Float32 — exact bits
- SwiGLU Adversarial 1-hop × all 34 dtypes — spectrum
- Sequential Adversarial 1-hop Float32 — exact bits
- Sequential Adversarial 1-hop × all 34 dtypes — spectrum
- Residual Adversarial 1-hop Float32 — exact bits
- Residual Adversarial 1-hop × all 34 dtypes — spectrum
- Mamba Adversarial 1-hop Float32 — exact bits
- Mamba Adversarial 1-hop × all 34 dtypes — spectrum
- Metacognition Adversarial 1-hop Float32 — exact bits
- Metacognition Adversarial 1-hop × all 34 dtypes — spectrum
- KMeans Adversarial 1-hop Float32 — exact bits
- KMeans Adversarial 1-hop × all 34 dtypes — spectrum
- GDN Adversarial 1-hop Float32 — exact bits
- NestedStack Adversarial 1-hop Float32 — exact bits
- NestedStack Adversarial 1-hop × all 34 dtypes — spectrum
- Embedding Adversarial 1-hop Float32 — exact bits
- Embedding Adversarial 1-hop × all 34 dtypes — spectrum
- MHA Adversarial 1-hop Float32 — exact bits
- MHA Adversarial 1-hop × all 34 dtypes — spectrum
- CNN1 Adversarial 1-hop Float32 — exact bits
- CNN1 Adversarial 1-hop × all 34 dtypes — spectrum
- ConvT1 Adversarial 1-hop Float32 — exact bits
- ConvT1 Adversarial 1-hop × all 34 dtypes — spectrum
- RNN Adversarial 1-hop Float32 — exact bits
- RNN Adversarial 1-hop × all 34 dtypes — spectrum
- LSTM Adversarial 1-hop Float32 — exact bits
- LSTM Adversarial 1-hop × all 34 dtypes — spectrum
- Dense Adversarial volume 1×1×1×1 Float32 — exact bits
- Dense Adversarial volume 2×1×1×1 chain Float32 — exact bits
- Dense Adversarial volume 2×2×2×1 Float32 — exact bits
- RMSNorm Adversarial volume 2×2×2×1 Float32 — exact bits
- LayerNorm Adversarial volume 2×2×2×1 Float32 — exact bits
- Softmax Adversarial volume 2×2×2×1 Float32 — exact bits
- SwiGLU Adversarial volume 2×2×2×1 Float32 — exact bits
- Sequential Adversarial volume 2×2×2×1 Float32 — exact bits
- Residual Adversarial volume 2×2×2×1 Float32 — exact bits
- Mamba Adversarial volume 2×2×2×1 Float32 — exact bits
- Metacognition Adversarial volume 2×2×2×1 Float32 — exact bits
- KMeans Adversarial volume 2×2×2×1 Float32 — exact bits
- GDN Adversarial volume 2×2×2×1 Float32 — exact bits
- NestedStack Adversarial volume 2×2×2×1 Float32 — exact bits
- Embedding Adversarial volume 2×2×2×1 Float32 — exact bits
- MHA Adversarial volume 2×2×2×1 Float32 — exact bits
- CNN1 Adversarial volume 2×2×2×1 Float32 — exact bits
- CNN2 Adversarial volume 2×2×2×1 Float32 — exact bits
- CNN3 Adversarial volume 2×2×2×1 Float32 — exact bits
- ConvT1 Adversarial volume 2×2×2×1 Float32 — exact bits
- ConvT2 Adversarial volume 2×2×2×1 Float32 — exact bits
- ConvT3 Adversarial volume 2×2×2×1 Float32 — exact bits
- RNN Adversarial volume 2×2×2×1 Float32 — exact bits
- LSTM Adversarial volume 2×2×2×1 Float32 — exact bits

#### `cnn.go` (6)

- CNN1 Float32 Linear pad — exact bits
- CNN2 Float32 ReLU — exact bits
- CNN3 Float32 Linear — exact bits
- CNN1 FormatNone × all 34 dtypes — spectrum + speed
- CNN2 FormatNone × all 34 dtypes — spectrum + speed
- CNN3 FormatNone × all 34 dtypes — spectrum + speed

#### `convt.go` (6)

- ConvT1 Float32 Linear pad — exact bits
- ConvT2 Float32 ReLU — exact bits
- ConvT3 Float32 Linear out_pad — exact bits
- ConvT1 FormatNone × all 34 dtypes — spectrum + speed
- ConvT2 FormatNone × all 34 dtypes — spectrum + speed
- ConvT3 FormatNone × all 34 dtypes — spectrum + speed

#### `dense_train.go` (7)

- Dense NormalBP leaf Float32 ReLU — exact bits (dX/dW/W/loss)
- Dense NormalBP leaf × all 34 dtypes — spectrum + speed
- Dense NormalBP volume 1×1×1×1 Float32 — exact bits
- Dense NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Dense NormalBP volume 2×2×2×1 Float32 — exact bits
- Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `embedding.go` (2)

- Embedding Float32 Linear — exact bits
- Embedding FormatNone × all 34 dtypes — spectrum + speed

#### `embedding_train.go` (7)

- Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Embedding NormalBP leaf × all 34 dtypes — spectrum + speed
- Embedding NormalBP volume 1×1×1×1 Float32 — exact bits
- Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Embedding NormalBP volume 2×2×2×1 Float32 — exact bits
- Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `freeze_train.go` (63)

- Parallel Freeze∥NormalBP Float32 — exact bits
- Parallel parent ModeFreeze Float32 — exact bits (all W frozen)
- Parallel Freeze∥NormalBP × all 34 dtypes — spectrum
- Parallel Freeze volume 2×2×2×1 Float32 — exact bits
- Dense Freeze multi-hop Float32 — exact bits (W/loss)
- Dense Freeze 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Freeze 1-hop Float32 — exact bits
- RMSNorm Freeze 1-hop × all 34 dtypes — spectrum
- LayerNorm Freeze 1-hop Float32 — exact bits
- LayerNorm Freeze 1-hop × all 34 dtypes — spectrum
- Softmax Freeze 1-hop Float32 — exact bits (loss)
- Softmax Freeze 1-hop × all 34 dtypes — spectrum
- SwiGLU Freeze 1-hop Float32 — exact bits
- SwiGLU Freeze 1-hop × all 34 dtypes — spectrum
- Sequential Freeze 1-hop Float32 — exact bits
- Sequential Freeze 1-hop × all 34 dtypes — spectrum
- Residual Freeze 1-hop Float32 — exact bits
- Residual Freeze 1-hop × all 34 dtypes — spectrum
- Mamba Freeze 1-hop Float32 — exact bits
- Mamba Freeze 1-hop × all 34 dtypes — spectrum
- Metacognition Freeze 1-hop Float32 — exact bits
- Metacognition Freeze 1-hop × all 34 dtypes — spectrum
- KMeans Freeze 1-hop Float32 — exact bits
- KMeans Freeze 1-hop × all 34 dtypes — spectrum
- GDN Freeze 1-hop Float32 — exact bits
- NestedStack Freeze 1-hop Float32 — exact bits
- NestedStack Freeze 1-hop × all 34 dtypes — spectrum
- Embedding Freeze 1-hop Float32 — exact bits
- Embedding Freeze 1-hop × all 34 dtypes — spectrum
- MHA Freeze 1-hop Float32 — exact bits
- MHA Freeze 1-hop × all 34 dtypes — spectrum
- CNN1 Freeze 1-hop Float32 — exact bits
- CNN1 Freeze 1-hop × all 34 dtypes — spectrum
- ConvT1 Freeze 1-hop Float32 — exact bits
- ConvT1 Freeze 1-hop × all 34 dtypes — spectrum
- RNN Freeze 1-hop Float32 — exact bits
- RNN Freeze 1-hop × all 34 dtypes — spectrum
- LSTM Freeze 1-hop Float32 — exact bits
- LSTM Freeze 1-hop × all 34 dtypes — spectrum
- Dense Freeze volume 1×1×1×1 Float32 — exact bits
- Dense Freeze volume 2×1×1×1 chain Float32 — exact bits
- Dense Freeze volume 2×2×2×1 Float32 — exact bits
- RMSNorm Freeze volume 2×2×2×1 Float32 — exact bits
- LayerNorm Freeze volume 2×2×2×1 Float32 — exact bits
- Softmax Freeze volume 2×2×2×1 Float32 — exact bits
- SwiGLU Freeze volume 2×2×2×1 Float32 — exact bits
- Sequential Freeze volume 2×2×2×1 Float32 — exact bits
- Residual Freeze volume 2×2×2×1 Float32 — exact bits
- Mamba Freeze volume 2×2×2×1 Float32 — exact bits
- Metacognition Freeze volume 2×2×2×1 Float32 — exact bits
- KMeans Freeze volume 2×2×2×1 Float32 — exact bits
- GDN Freeze volume 2×2×2×1 Float32 — exact bits
- NestedStack Freeze volume 2×2×2×1 Float32 — exact bits
- Embedding Freeze volume 2×2×2×1 Float32 — exact bits
- MHA Freeze volume 2×2×2×1 Float32 — exact bits
- CNN1 Freeze volume 2×2×2×1 Float32 — exact bits
- CNN2 Freeze volume 2×2×2×1 Float32 — exact bits
- CNN3 Freeze volume 2×2×2×1 Float32 — exact bits
- ConvT1 Freeze volume 2×2×2×1 Float32 — exact bits
- ConvT2 Freeze volume 2×2×2×1 Float32 — exact bits
- ConvT3 Freeze volume 2×2×2×1 Float32 — exact bits
- RNN Freeze volume 2×2×2×1 Float32 — exact bits
- LSTM Freeze volume 2×2×2×1 Float32 — exact bits

#### `gdn.go` (3)

- GDN Float32 primary — exact bits
- GDN Float32 alt GQA — exact bits
- GDN Float32 spectrum + speed

#### `gdn_train.go` (4)

- GDN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- GDN NormalBP volume 1×1×1×1 Float32 — exact bits
- GDN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- GDN NormalBP volume 2×2×2×1 Float32 — exact bits

#### `kmeans.go` (3)

- KMeans probabilities Float32 Linear — exact bits
- KMeans features Float32 Linear — exact bits
- KMeans both modes × all 34 dtypes — spectrum + speed

#### `kmeans_train.go` (8)

- KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed
- KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits
- KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `layernorm.go` (4)

- LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits
- LayerNorm Float32 [batch,seq,dim] random affine — exact bits
- LayerNorm Float32 custom eps — exact bits
- LayerNorm FormatNone × all 34 dtypes — spectrum + speed

#### `layernorm_train.go` (7)

- LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)
- LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `lstm.go` (2)

- LSTM Float32 — exact bits
- LSTM FormatNone × all 34 dtypes — spectrum + speed

#### `lstm_train.go` (7)

- LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- LSTM NormalBP leaf × all 34 dtypes — spectrum + speed
- LSTM NormalBP volume 1×1×1×1 Float32 — exact bits
- LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LSTM NormalBP volume 2×2×2×1 Float32 — exact bits
- LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mamba.go` (3)

- Mamba Float32 [B,T,D] — exact bits
- Mamba Float32 alt shape — exact bits
- Mamba × all 34 dtypes — spectrum + speed

#### `mamba_train.go` (7)

- Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Mamba NormalBP leaf × all 34 dtypes — spectrum + speed
- Mamba NormalBP volume 1×1×1×1 Float32 — exact bits
- Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Mamba NormalBP volume 2×2×2×1 Float32 — exact bits
- Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `memory_train.go` (63)

- Parallel Memory gate open (thresh=0) Float32 — exact bits
- Parallel Memory gate closed (high thresh) Float32 — exact bits
- Parallel Memory gate open × all 34 dtypes — spectrum
- Parallel Memory volume 2×2×2×1 Float32 — exact bits (≡ FamilyBP)
- Dense Memory multi-hop Float32 — exact bits (W/loss)
- Dense Memory 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Memory 1-hop Float32 — exact bits
- RMSNorm Memory 1-hop × all 34 dtypes — spectrum
- LayerNorm Memory 1-hop Float32 — exact bits
- LayerNorm Memory 1-hop × all 34 dtypes — spectrum
- Softmax Memory 1-hop Float32 — exact bits (loss)
- Softmax Memory 1-hop × all 34 dtypes — spectrum
- SwiGLU Memory 1-hop Float32 — exact bits
- SwiGLU Memory 1-hop × all 34 dtypes — spectrum
- Sequential Memory 1-hop Float32 — exact bits
- Sequential Memory 1-hop × all 34 dtypes — spectrum
- Residual Memory 1-hop Float32 — exact bits
- Residual Memory 1-hop × all 34 dtypes — spectrum
- Mamba Memory 1-hop Float32 — exact bits
- Mamba Memory 1-hop × all 34 dtypes — spectrum
- Metacognition Memory 1-hop Float32 — exact bits
- Metacognition Memory 1-hop × all 34 dtypes — spectrum
- KMeans Memory 1-hop Float32 — exact bits
- KMeans Memory 1-hop × all 34 dtypes — spectrum
- GDN Memory 1-hop Float32 — exact bits
- NestedStack Memory 1-hop Float32 — exact bits
- NestedStack Memory 1-hop × all 34 dtypes — spectrum
- Embedding Memory 1-hop Float32 — exact bits
- Embedding Memory 1-hop × all 34 dtypes — spectrum
- MHA Memory 1-hop Float32 — exact bits
- MHA Memory 1-hop × all 34 dtypes — spectrum
- CNN1 Memory 1-hop Float32 — exact bits
- CNN1 Memory 1-hop × all 34 dtypes — spectrum
- ConvT1 Memory 1-hop Float32 — exact bits
- ConvT1 Memory 1-hop × all 34 dtypes — spectrum
- RNN Memory 1-hop Float32 — exact bits
- RNN Memory 1-hop × all 34 dtypes — spectrum
- LSTM Memory 1-hop Float32 — exact bits
- LSTM Memory 1-hop × all 34 dtypes — spectrum
- Dense Memory volume 1×1×1×1 Float32 — exact bits
- Dense Memory volume 2×1×1×1 chain Float32 — exact bits
- Dense Memory volume 2×2×2×1 Float32 — exact bits
- RMSNorm Memory volume 2×2×2×1 Float32 — exact bits
- LayerNorm Memory volume 2×2×2×1 Float32 — exact bits
- Softmax Memory volume 2×2×2×1 Float32 — exact bits
- SwiGLU Memory volume 2×2×2×1 Float32 — exact bits
- Sequential Memory volume 2×2×2×1 Float32 — exact bits
- Residual Memory volume 2×2×2×1 Float32 — exact bits
- Mamba Memory volume 2×2×2×1 Float32 — exact bits
- Metacognition Memory volume 2×2×2×1 Float32 — exact bits
- KMeans Memory volume 2×2×2×1 Float32 — exact bits
- GDN Memory volume 2×2×2×1 Float32 — exact bits
- NestedStack Memory volume 2×2×2×1 Float32 — exact bits
- Embedding Memory volume 2×2×2×1 Float32 — exact bits
- MHA Memory volume 2×2×2×1 Float32 — exact bits
- CNN1 Memory volume 2×2×2×1 Float32 — exact bits
- CNN2 Memory volume 2×2×2×1 Float32 — exact bits
- CNN3 Memory volume 2×2×2×1 Float32 — exact bits
- ConvT1 Memory volume 2×2×2×1 Float32 — exact bits
- ConvT2 Memory volume 2×2×2×1 Float32 — exact bits
- ConvT3 Memory volume 2×2×2×1 Float32 — exact bits
- RNN Memory volume 2×2×2×1 Float32 — exact bits
- LSTM Memory volume 2×2×2×1 Float32 — exact bits

#### `meshtween_train.go` (28)

- Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTween volume 1×1×1×1 Float32 — exact bits
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTween volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTween volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTween volume 2×2×2×1 Float32 — exact bits
- Residual MeshTween volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTween volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTween volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTween volume 2×2×2×1 Float32 — exact bits
- GDN MeshTween volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTween volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTween volume 2×2×2×1 Float32 — exact bits
- MHA MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact bits
- RNN MeshTween volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTween volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTween volume 2×2×2×1 Float32 — exact bits

#### `meshtweenalt_train.go` (28)

- Dense MeshTweenAlt mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTweenAlt mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTweenAlt mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTweenAlt volume 2×2×2×1 Float32 — exact bits

#### `meshtweenchain_train.go` (28)

- Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTweenChain volume 2×2×2×1 Float32 — exact bits

#### `meshtweensplit_train.go` (28)

- Dense MeshTweenSplit mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplit mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTweenSplit mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTweenSplit volume 2×2×2×1 Float32 — exact bits

#### `meshtweensplitfastproxy_train.go` (28)

- Dense MeshTweenSplitFastProxy mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitFastProxy mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitFastProxy mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTweenSplitFastProxy volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitFastProxy volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits

#### `meshtweensplitsparse_train.go` (28)

- Dense MeshTweenSplitSparse mesh-clock 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitSparse mesh-clock 2×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitSparse mesh-clock 2×2×2×1 Float32 — exact bits
- Dense MeshTweenSplitSparse volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplitSparse volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Parallel MeshTweenSplitSparse volume 2×2×2×1 Float32 — exact bits

#### `metacognition.go` (3)

- Metacognition Float32 [B,D] — exact bits
- Metacognition Float32 [B,T,D] — exact bits
- Metacognition × all 34 dtypes — spectrum + speed

#### `metacognition_train.go` (7)

- Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed
- Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits
- Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mha.go` (4)

- MHA DecoderCausal Float32 smoke — exact bits
- MHA GQA 4h/2kv Float32 — exact bits
- MHA Encoder bidirectional Float32 — exact bits
- MHA DecoderCausal FormatNone × all 34 dtypes — spectrum + speed

#### `mha_train.go` (8)

- MHA NormalBP leaf DecoderCausal Float32 — exact bits
- MHA NormalBP leaf GQA + Encoder Float32 — exact bits
- MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed
- MHA NormalBP volume 1×1×1×1 Float32 — exact bits
- MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits
- MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- MHA NormalBP volume 2×2×2×1 Float32 — exact bits
- MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `parallel.go` (9)

- Parallel Add Float32 [B,D] — exact bits
- Parallel Add Float32 [B,T,D] — exact bits
- Parallel Concat Float32 — exact bits
- Parallel Avg Float32 — exact bits
- Parallel Filter Float32 — exact bits
- Parallel Max Float32 — exact bits
- Parallel SparseK Float32 — exact bits
- Parallel Disagree Float32 (2+3 branches) — exact bits
- Parallel Add × all 34 dtypes — spectrum + speed

#### `parallel_nbp_train.go` (9)

- Dense Add Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP × all 34 dtypes — spectrum
- Dense Avg Parallel NormalBP Float32 — exact bits
- Dense Concat Parallel NormalBP Float32 — exact bits
- Dense Max Parallel NormalBP Float32 — exact bits
- Dense SparseK Parallel NormalBP Float32 — exact bits
- Dense Disagree Parallel NormalBP Float32 — exact bits
- Dense Filter Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits

#### `parallel_tween_train.go` (16)

- Dense Add Parallel Tween Float32 — exact bits
- Dense Add Parallel Tween × all 34 dtypes — spectrum
- Dense Avg Parallel Tween Float32 — exact bits
- Dense Add Parallel Tween volume 2×2×2×1 Float32 — exact bits
- Dense Add Parallel TweenChain Float32 — exact bits
- Dense Add Parallel TweenChain × all 34 dtypes — spectrum
- Dense Avg Parallel TweenChain Float32 — exact bits
- Dense Add Parallel TweenChain volume 2×2×2×1 Float32 — exact bits
- Dense Add Parallel TweenSplit Float32 — exact bits
- Dense Add Parallel TweenSplit × all 34 dtypes — spectrum
- Dense Avg Parallel TweenSplit Float32 — exact bits
- Dense Add Parallel TweenSplit volume 2×2×2×1 Float32 — exact bits
- Dense Add Parallel TweenAlt Float32 — exact bits
- Dense Add Parallel TweenAlt × all 34 dtypes — spectrum
- Dense Avg Parallel TweenAlt Float32 — exact bits
- Dense Add Parallel TweenAlt volume 2×2×2×1 Float32 — exact bits

#### `residual.go` (3)

- Residual Dense Float32 [B,D] — exact bits
- Residual Dense Float32 [B,T,D] — exact bits
- Residual Dense × all 34 dtypes — spectrum + speed

#### `residual_train.go` (7)

- Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Residual NormalBP leaf × all 34 dtypes — spectrum + speed
- Residual NormalBP volume 1×1×1×1 Float32 — exact bits
- Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Residual NormalBP volume 2×2×2×1 Float32 — exact bits
- Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rmsnorm.go` (4)

- RMSNorm Float32 [batch,dim] γ=1 — exact bits
- RMSNorm Float32 [batch,seq,dim] random γ — exact bits
- RMSNorm Float32 custom eps — exact bits
- RMSNorm FormatNone × all 34 dtypes — spectrum + speed

#### `rmsnorm_train.go` (7)

- RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)
- RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rnn.go` (2)

- RNN Float32 — exact bits
- RNN FormatNone × all 34 dtypes — spectrum + speed

#### `rnn_train.go` (7)

- RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- RNN NormalBP leaf × all 34 dtypes — spectrum + speed
- RNN NormalBP volume 1×1×1×1 Float32 — exact bits
- RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RNN NormalBP volume 2×2×2×1 Float32 — exact bits
- RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `seqmix.go` (3)

- seqmix Kind iota + String — Go
- seqmix mixer tags — Go MHA/Mamba/GDN
- seqmix Kind names + tags — Go vs C++ WSX1

#### `sequential.go` (3)

- Sequential Dense Float32 [B,D] — exact bits
- Sequential Dense Float32 [B,T,D] — exact bits
- Sequential Dense × all 34 dtypes — spectrum + speed

#### `sequential_train.go` (7)

- Sequential NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Sequential NormalBP leaf × all 34 dtypes — spectrum + speed
- Sequential NormalBP volume 1×1×1×1 Float32 — exact bits
- Sequential NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Sequential NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Sequential NormalBP volume 2×2×2×1 Float32 — exact bits
- Sequential NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `shadow_train.go` (63)

- Parallel Shadow∥NormalBP KD Float32 — exact bits
- Parallel Shadow∥NormalBP KD coef=0.5 Float32 — exact bits
- Parallel Shadow∥NormalBP KD × all 34 dtypes — spectrum
- Parallel Shadow volume 2×2×2×1 Float32 — exact bits (≡ Freeze)
- Dense Shadow multi-hop Float32 — exact bits (W/loss)
- Dense Shadow 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Shadow 1-hop Float32 — exact bits
- RMSNorm Shadow 1-hop × all 34 dtypes — spectrum
- LayerNorm Shadow 1-hop Float32 — exact bits
- LayerNorm Shadow 1-hop × all 34 dtypes — spectrum
- Softmax Shadow 1-hop Float32 — exact bits (loss)
- Softmax Shadow 1-hop × all 34 dtypes — spectrum
- SwiGLU Shadow 1-hop Float32 — exact bits
- SwiGLU Shadow 1-hop × all 34 dtypes — spectrum
- Sequential Shadow 1-hop Float32 — exact bits
- Sequential Shadow 1-hop × all 34 dtypes — spectrum
- Residual Shadow 1-hop Float32 — exact bits
- Residual Shadow 1-hop × all 34 dtypes — spectrum
- Mamba Shadow 1-hop Float32 — exact bits
- Mamba Shadow 1-hop × all 34 dtypes — spectrum
- Metacognition Shadow 1-hop Float32 — exact bits
- Metacognition Shadow 1-hop × all 34 dtypes — spectrum
- KMeans Shadow 1-hop Float32 — exact bits
- KMeans Shadow 1-hop × all 34 dtypes — spectrum
- GDN Shadow 1-hop Float32 — exact bits
- NestedStack Shadow 1-hop Float32 — exact bits
- NestedStack Shadow 1-hop × all 34 dtypes — spectrum
- Embedding Shadow 1-hop Float32 — exact bits
- Embedding Shadow 1-hop × all 34 dtypes — spectrum
- MHA Shadow 1-hop Float32 — exact bits
- MHA Shadow 1-hop × all 34 dtypes — spectrum
- CNN1 Shadow 1-hop Float32 — exact bits
- CNN1 Shadow 1-hop × all 34 dtypes — spectrum
- ConvT1 Shadow 1-hop Float32 — exact bits
- ConvT1 Shadow 1-hop × all 34 dtypes — spectrum
- RNN Shadow 1-hop Float32 — exact bits
- RNN Shadow 1-hop × all 34 dtypes — spectrum
- LSTM Shadow 1-hop Float32 — exact bits
- LSTM Shadow 1-hop × all 34 dtypes — spectrum
- Dense Shadow volume 1×1×1×1 Float32 — exact bits
- Dense Shadow volume 2×1×1×1 chain Float32 — exact bits
- Dense Shadow volume 2×2×2×1 Float32 — exact bits
- RMSNorm Shadow volume 2×2×2×1 Float32 — exact bits
- LayerNorm Shadow volume 2×2×2×1 Float32 — exact bits
- Softmax Shadow volume 2×2×2×1 Float32 — exact bits
- SwiGLU Shadow volume 2×2×2×1 Float32 — exact bits
- Sequential Shadow volume 2×2×2×1 Float32 — exact bits
- Residual Shadow volume 2×2×2×1 Float32 — exact bits
- Mamba Shadow volume 2×2×2×1 Float32 — exact bits
- Metacognition Shadow volume 2×2×2×1 Float32 — exact bits
- KMeans Shadow volume 2×2×2×1 Float32 — exact bits
- GDN Shadow volume 2×2×2×1 Float32 — exact bits
- NestedStack Shadow volume 2×2×2×1 Float32 — exact bits
- Embedding Shadow volume 2×2×2×1 Float32 — exact bits
- MHA Shadow volume 2×2×2×1 Float32 — exact bits
- CNN1 Shadow volume 2×2×2×1 Float32 — exact bits
- CNN2 Shadow volume 2×2×2×1 Float32 — exact bits
- CNN3 Shadow volume 2×2×2×1 Float32 — exact bits
- ConvT1 Shadow volume 2×2×2×1 Float32 — exact bits
- ConvT2 Shadow volume 2×2×2×1 Float32 — exact bits
- ConvT3 Shadow volume 2×2×2×1 Float32 — exact bits
- RNN Shadow volume 2×2×2×1 Float32 — exact bits
- LSTM Shadow volume 2×2×2×1 Float32 — exact bits

#### `softmax.go` (9)

- Softmax Standard Float32 — exact bits
- Softmax Temperature Float32 — exact bits
- Softmax Masked Float32 — exact bits
- Softmax Sparse Float32 — exact bits
- Softmax Entmax Float32 — exact bits
- Softmax Grid Float32 — exact bits
- Softmax Hierarchical Float32 — exact bits
- Softmax Gumbel Float32 — smoke (non-deterministic)
- Softmax all Kind × all 34 dtypes — spectrum + speed

#### `softmax_train.go` (9)

- Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)
- Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)
- Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed
- Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum
- Softmax NormalBP volume 1×1×1×1 Float32 — exact bits
- Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Softmax NormalBP volume 2×2×2×1 Float32 — exact bits
- Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stack_train.go` (7)

- Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Stack NormalBP leaf × all 34 dtypes — spectrum + speed
- Stack NormalBP volume 1×1×1×1 Float32 — exact bits
- Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Stack NormalBP volume 2×2×2×1 Float32 — exact bits
- Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stepbp_train.go` (61)

- Dense StepBP multi-hop Float32 — exact bits (W/loss)
- Dense StepBP 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepBP 1-hop Float32 — exact bits
- RMSNorm StepBP 1-hop × all 34 dtypes — spectrum
- LayerNorm StepBP 1-hop Float32 — exact bits
- LayerNorm StepBP 1-hop × all 34 dtypes — spectrum
- Softmax StepBP 1-hop Float32 — exact bits (loss)
- Softmax StepBP 1-hop × all 34 dtypes — spectrum
- SwiGLU StepBP 1-hop Float32 — exact bits
- SwiGLU StepBP 1-hop × all 34 dtypes — spectrum
- Sequential StepBP 1-hop Float32 — exact bits
- Sequential StepBP 1-hop × all 34 dtypes — spectrum
- Residual StepBP 1-hop Float32 — exact bits
- Residual StepBP 1-hop × all 34 dtypes — spectrum
- Mamba StepBP 1-hop Float32 — exact bits
- Mamba StepBP 1-hop × all 34 dtypes — spectrum
- Metacognition StepBP 1-hop Float32 — exact bits
- Metacognition StepBP 1-hop × all 34 dtypes — spectrum
- KMeans StepBP 1-hop Float32 — exact bits
- KMeans StepBP 1-hop × all 34 dtypes — spectrum
- GDN StepBP 1-hop Float32 — exact bits
- NestedStack StepBP 1-hop Float32 — exact bits
- NestedStack StepBP 1-hop × all 34 dtypes — spectrum
- Embedding StepBP 1-hop Float32 — exact bits
- Embedding StepBP 1-hop × all 34 dtypes — spectrum
- MHA StepBP 1-hop Float32 — exact bits
- MHA StepBP 1-hop × all 34 dtypes — spectrum
- CNN1 StepBP 1-hop Float32 — exact bits
- CNN1 StepBP 1-hop × all 34 dtypes — spectrum
- ConvT1 StepBP 1-hop Float32 — exact bits
- ConvT1 StepBP 1-hop × all 34 dtypes — spectrum
- RNN StepBP 1-hop Float32 — exact bits
- RNN StepBP 1-hop × all 34 dtypes — spectrum
- LSTM StepBP 1-hop Float32 — exact bits
- LSTM StepBP 1-hop × all 34 dtypes — spectrum
- Dense MeshBP 1×1×1×1 Float32 — exact bits
- Dense MeshBP 2×1×1×1 chain Float32 — exact bits
- Dense MeshBP 2×2×2×1 Float32 — exact bits
- Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact bits
- RMSNorm MeshBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshBP volume 2×2×2×1 Float32 — exact bits
- Softmax MeshBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshBP volume 2×2×2×1 Float32 — exact bits
- Sequential MeshBP volume 2×2×2×1 Float32 — exact bits
- Residual MeshBP volume 2×2×2×1 Float32 — exact bits
- Mamba MeshBP volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshBP volume 2×2×2×1 Float32 — exact bits
- KMeans MeshBP volume 2×2×2×1 Float32 — exact bits
- GDN MeshBP volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshBP volume 2×2×2×1 Float32 — exact bits
- Embedding MeshBP volume 2×2×2×1 Float32 — exact bits
- MHA MeshBP volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshBP volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshBP volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshBP volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshBP volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshBP volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshBP volume 2×2×2×1 Float32 — exact bits
- RNN MeshBP volume 2×2×2×1 Float32 — exact bits
- LSTM MeshBP volume 2×2×2×1 Float32 — exact bits
- Parallel MeshBP volume 2×2×2×1 Float32 — exact bits

#### `steptween_train.go` (48)

- Dense StepTween multi-hop Float32 — exact bits (W/loss)
- Dense StepTween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTween 1-hop Float32 — exact bits
- RMSNorm StepTween 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTween 1-hop Float32 — exact bits
- LayerNorm StepTween 1-hop × all 34 dtypes — spectrum
- Softmax StepTween 1-hop Float32 — exact bits (loss)
- Softmax StepTween 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTween 1-hop Float32 — exact bits
- SwiGLU StepTween 1-hop × all 34 dtypes — spectrum
- Sequential StepTween 1-hop Float32 — exact bits
- Sequential StepTween 1-hop × all 34 dtypes — spectrum
- Residual StepTween 1-hop Float32 — exact bits
- Residual StepTween 1-hop × all 34 dtypes — spectrum
- Mamba StepTween 1-hop Float32 — exact bits
- Mamba StepTween 1-hop × all 34 dtypes — spectrum
- Embedding StepTween 1-hop Float32 — exact bits
- Embedding StepTween 1-hop × all 34 dtypes — spectrum
- MHA StepTween 1-hop Float32 — exact bits
- MHA StepTween 1-hop × all 34 dtypes — spectrum
- CNN1 StepTween 1-hop Float32 — exact bits
- CNN1 StepTween 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTween 1-hop Float32 — exact bits
- ConvT1 StepTween 1-hop × all 34 dtypes — spectrum
- RNN StepTween 1-hop Float32 — exact bits
- RNN StepTween 1-hop × all 34 dtypes — spectrum
- LSTM StepTween 1-hop Float32 — exact bits
- LSTM StepTween 1-hop × all 34 dtypes — spectrum
- Dense MeshTween volume 1×1×1×1 Float32 — exact bits
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTween volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTween volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTween volume 2×2×2×1 Float32 — exact bits
- Residual MeshTween volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTween volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTween volume 2×2×2×1 Float32 — exact bits
- MHA MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact bits
- RNN MeshTween volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTween volume 2×2×2×1 Float32 — exact bits

#### `steptweenalt_train.go` (59)

- Dense StepTweenAlt multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenAlt 1-hop Float32 — exact bits
- RMSNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenAlt 1-hop Float32 — exact bits
- LayerNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenAlt 1-hop Float32 — exact bits (loss)
- Softmax StepTweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenAlt 1-hop Float32 — exact bits
- SwiGLU StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenAlt 1-hop Float32 — exact bits
- Sequential StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Residual StepTweenAlt 1-hop Float32 — exact bits
- Residual StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenAlt 1-hop Float32 — exact bits
- Mamba StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenAlt 1-hop Float32 — exact bits
- Metacognition StepTweenAlt 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenAlt 1-hop Float32 — exact bits
- KMeans StepTweenAlt 1-hop × all 34 dtypes — spectrum
- GDN StepTweenAlt 1-hop Float32 — exact bits
- NestedStack StepTweenAlt 1-hop Float32 — exact bits
- NestedStack StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenAlt 1-hop Float32 — exact bits
- Embedding StepTweenAlt 1-hop × all 34 dtypes — spectrum
- MHA StepTweenAlt 1-hop Float32 — exact bits
- MHA StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenAlt 1-hop Float32 — exact bits
- CNN1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenAlt 1-hop Float32 — exact bits
- ConvT1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- RNN StepTweenAlt 1-hop Float32 — exact bits
- RNN StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenAlt 1-hop Float32 — exact bits
- LSTM StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact bits

#### `steptweenchain_train.go` (59)

- Dense StepTweenChain multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenChain 1-hop Float32 — exact bits
- RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenChain 1-hop Float32 — exact bits
- LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenChain 1-hop Float32 — exact bits (loss)
- Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenChain 1-hop Float32 — exact bits
- SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenChain 1-hop Float32 — exact bits
- Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum
- Residual StepTweenChain 1-hop Float32 — exact bits
- Residual StepTweenChain 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenChain 1-hop Float32 — exact bits
- Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenChain 1-hop Float32 — exact bits
- Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenChain 1-hop Float32 — exact bits
- KMeans StepTweenChain 1-hop × all 34 dtypes — spectrum
- GDN StepTweenChain 1-hop Float32 — exact bits
- NestedStack StepTweenChain 1-hop Float32 — exact bits
- NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenChain 1-hop Float32 — exact bits
- Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum
- MHA StepTweenChain 1-hop Float32 — exact bits
- MHA StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenChain 1-hop Float32 — exact bits
- CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenChain 1-hop Float32 — exact bits
- ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- RNN StepTweenChain 1-hop Float32 — exact bits
- RNN StepTweenChain 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenChain 1-hop Float32 — exact bits
- LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact bits

#### `steptweensplit_train.go` (59)

- Dense StepTweenSplit multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplit 1-hop Float32 — exact bits
- RMSNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplit 1-hop Float32 — exact bits
- LayerNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplit 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplit 1-hop Float32 — exact bits
- SwiGLU StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplit 1-hop Float32 — exact bits
- Sequential StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplit 1-hop Float32 — exact bits
- Residual StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplit 1-hop Float32 — exact bits
- Mamba StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplit 1-hop Float32 — exact bits
- Metacognition StepTweenSplit 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplit 1-hop Float32 — exact bits
- KMeans StepTweenSplit 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplit 1-hop Float32 — exact bits
- NestedStack StepTweenSplit 1-hop Float32 — exact bits
- NestedStack StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplit 1-hop Float32 — exact bits
- Embedding StepTweenSplit 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplit 1-hop Float32 — exact bits
- MHA StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplit 1-hop Float32 — exact bits
- CNN1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplit 1-hop Float32 — exact bits
- ConvT1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplit 1-hop Float32 — exact bits
- RNN StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplit 1-hop Float32 — exact bits
- LSTM StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact bits
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact bits
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- KMeans MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact bits
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitfastproxy_train.go` (59)

- Dense StepTweenSplitFastProxy multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitFastProxy 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitFastProxy 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitFastProxy 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitFastProxy 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitFastProxy 1-hop Float32 — exact bits
- Sequential StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitFastProxy 1-hop Float32 — exact bits
- Residual StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitFastProxy 1-hop Float32 — exact bits
- Mamba StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitFastProxy 1-hop Float32 — exact bits
- Metacognition StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitFastProxy 1-hop Float32 — exact bits
- KMeans StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitFastProxy 1-hop Float32 — exact bits
- NestedStack StepTweenSplitFastProxy 1-hop Float32 — exact bits
- NestedStack StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitFastProxy 1-hop Float32 — exact bits
- Embedding StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitFastProxy 1-hop Float32 — exact bits
- MHA StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitFastProxy 1-hop Float32 — exact bits
- CNN1 StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitFastProxy 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitFastProxy 1-hop Float32 — exact bits
- RNN StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitFastProxy 1-hop Float32 — exact bits
- LSTM StepTweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitFastProxy volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitFastProxy volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitheadproxy_train.go` (59)

- Dense StepTweenSplitHeadProxy multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitHeadProxy 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- Sequential StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- Residual StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- Mamba StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- Metacognition StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- KMeans StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- NestedStack StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- NestedStack StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- Embedding StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- MHA StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- CNN1 StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- RNN StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitHeadProxy 1-hop Float32 — exact bits
- LSTM StepTweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitHeadProxy volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitHeadProxy volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitheadproxyasync_train.go` (59)

- Dense StepTweenSplitHeadProxyAsync multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Sequential StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Residual StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Mamba StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Metacognition StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- KMeans StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- NestedStack StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- NestedStack StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Embedding StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- MHA StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- CNN1 StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- RNN StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- LSTM StepTweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitHeadProxyAsync volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitHeadProxyAsync volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitlinear_train.go` (59)

- Dense StepTweenSplitLinear multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitLinear 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitLinear 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitLinear 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitLinear 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitLinear 1-hop Float32 — exact bits
- Sequential StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitLinear 1-hop Float32 — exact bits
- Residual StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitLinear 1-hop Float32 — exact bits
- Mamba StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitLinear 1-hop Float32 — exact bits
- Metacognition StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitLinear 1-hop Float32 — exact bits
- KMeans StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitLinear 1-hop Float32 — exact bits
- NestedStack StepTweenSplitLinear 1-hop Float32 — exact bits
- NestedStack StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitLinear 1-hop Float32 — exact bits
- Embedding StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitLinear 1-hop Float32 — exact bits
- MHA StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitLinear 1-hop Float32 — exact bits
- CNN1 StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitLinear 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitLinear 1-hop Float32 — exact bits
- RNN StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitLinear 1-hop Float32 — exact bits
- LSTM StepTweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitLinear volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitLinear volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitLinear volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitlinearcache_train.go` (59)

- Dense StepTweenSplitLinearCache multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitLinearCache 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitLinearCache 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitLinearCache 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitLinearCache 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitLinearCache 1-hop Float32 — exact bits
- Sequential StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitLinearCache 1-hop Float32 — exact bits
- Residual StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitLinearCache 1-hop Float32 — exact bits
- Mamba StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitLinearCache 1-hop Float32 — exact bits
- Metacognition StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitLinearCache 1-hop Float32 — exact bits
- KMeans StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitLinearCache 1-hop Float32 — exact bits
- NestedStack StepTweenSplitLinearCache 1-hop Float32 — exact bits
- NestedStack StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitLinearCache 1-hop Float32 — exact bits
- Embedding StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitLinearCache 1-hop Float32 — exact bits
- MHA StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitLinearCache 1-hop Float32 — exact bits
- CNN1 StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitLinearCache 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitLinearCache 1-hop Float32 — exact bits
- RNN StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitLinearCache 1-hop Float32 — exact bits
- LSTM StepTweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitLinearCache volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitLinearCache volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits

#### `steptweensplitsparse_train.go` (59)

- Dense StepTweenSplitSparse multi-hop Float32 — exact bits (W/loss)
- Dense StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplitSparse 1-hop Float32 — exact bits
- RMSNorm StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplitSparse 1-hop Float32 — exact bits
- LayerNorm StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplitSparse 1-hop Float32 — exact bits (loss)
- Softmax StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplitSparse 1-hop Float32 — exact bits
- SwiGLU StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplitSparse 1-hop Float32 — exact bits
- Sequential StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplitSparse 1-hop Float32 — exact bits
- Residual StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplitSparse 1-hop Float32 — exact bits
- Mamba StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplitSparse 1-hop Float32 — exact bits
- Metacognition StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- KMeans StepTweenSplitSparse 1-hop Float32 — exact bits
- KMeans StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- GDN StepTweenSplitSparse 1-hop Float32 — exact bits
- NestedStack StepTweenSplitSparse 1-hop Float32 — exact bits
- NestedStack StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplitSparse 1-hop Float32 — exact bits
- Embedding StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplitSparse 1-hop Float32 — exact bits
- MHA StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplitSparse 1-hop Float32 — exact bits
- CNN1 StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplitSparse 1-hop Float32 — exact bits
- ConvT1 StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplitSparse 1-hop Float32 — exact bits
- RNN StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplitSparse 1-hop Float32 — exact bits
- LSTM StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitSparse volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitSparse volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitSparse volume 2×2×2×1 Float32 — exact bits

#### `suite.go` (12)

- Locate welvet_dense_fwd binary
- Spectrum ladder self-check
- Dense forward identity+bias Float32 CPU — exact bits
- Dense forward ReLU Float32 CPU — exact bits
- CPU FormatNone × all 34 dtypes — spectrum + speed
- SIMD FormatNone × all 34 dtypes — Go SIMD vs C++ SIMD
- Cross: Go CPU vs C++ SIMD Float32 — spectrum report
- Architecture topology 2×3×4×2 + remote link — Go vs C++
- Volume Dense 1×1×1×2 chain — exact bits
- Volume Dense 2×1×1×1 remote hop — exact bits
- Volume Dense 3×3×3×1 × all 34 dtypes — spectrum + speed
- C++ gap census present (welvet_dense_fwd --gaps)

#### `swiglu.go` (4)

- SwiGLU Float32 [batch,in] smoke — exact bits
- SwiGLU Float32 [batch,seq,in] — exact bits
- SwiGLU DefaultFFN Float32 — exact bits
- SwiGLU FormatNone × all 34 dtypes — spectrum + speed

#### `swiglu_train.go` (7)

- SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed
- SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `tween_train.go` (59)

- Dense Tween multi-hop Float32 — exact bits (W/loss)
- Dense Tween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Tween 1-hop Float32 — exact bits
- RMSNorm Tween 1-hop × all 34 dtypes — spectrum
- LayerNorm Tween 1-hop Float32 — exact bits
- LayerNorm Tween 1-hop × all 34 dtypes — spectrum
- Softmax Tween 1-hop Float32 — exact bits (loss)
- Softmax Tween 1-hop × all 34 dtypes — spectrum
- SwiGLU Tween 1-hop Float32 — exact bits
- SwiGLU Tween 1-hop × all 34 dtypes — spectrum
- Sequential Tween 1-hop Float32 — exact bits
- Sequential Tween 1-hop × all 34 dtypes — spectrum
- Residual Tween 1-hop Float32 — exact bits
- Residual Tween 1-hop × all 34 dtypes — spectrum
- Mamba Tween 1-hop Float32 — exact bits
- Mamba Tween 1-hop × all 34 dtypes — spectrum
- Metacognition Tween 1-hop Float32 — exact bits
- Metacognition Tween 1-hop × all 34 dtypes — spectrum
- KMeans Tween 1-hop Float32 — exact bits
- KMeans Tween 1-hop × all 34 dtypes — spectrum
- GDN Tween 1-hop Float32 — exact bits
- NestedStack Tween 1-hop Float32 — exact bits
- NestedStack Tween 1-hop × all 34 dtypes — spectrum
- Embedding Tween 1-hop Float32 — exact bits
- Embedding Tween 1-hop × all 34 dtypes — spectrum
- MHA Tween 1-hop Float32 — exact bits
- MHA Tween 1-hop × all 34 dtypes — spectrum
- CNN1 Tween 1-hop Float32 — exact bits
- CNN1 Tween 1-hop × all 34 dtypes — spectrum
- ConvT1 Tween 1-hop Float32 — exact bits
- ConvT1 Tween 1-hop × all 34 dtypes — spectrum
- RNN Tween 1-hop Float32 — exact bits
- RNN Tween 1-hop × all 34 dtypes — spectrum
- LSTM Tween 1-hop Float32 — exact bits
- LSTM Tween 1-hop × all 34 dtypes — spectrum
- Dense Tween volume 1×1×1×1 Float32 — exact bits
- Dense Tween volume 2×1×1×1 chain Float32 — exact bits
- Dense Tween volume 2×2×2×1 Float32 — exact bits
- RMSNorm Tween volume 2×2×2×1 Float32 — exact bits
- LayerNorm Tween volume 2×2×2×1 Float32 — exact bits
- Softmax Tween volume 2×2×2×1 Float32 — exact bits
- SwiGLU Tween volume 2×2×2×1 Float32 — exact bits
- Sequential Tween volume 2×2×2×1 Float32 — exact bits
- Residual Tween volume 2×2×2×1 Float32 — exact bits
- Mamba Tween volume 2×2×2×1 Float32 — exact bits
- Metacognition Tween volume 2×2×2×1 Float32 — exact bits
- KMeans Tween volume 2×2×2×1 Float32 — exact bits
- GDN Tween volume 2×2×2×1 Float32 — exact bits
- NestedStack Tween volume 2×2×2×1 Float32 — exact bits
- Embedding Tween volume 2×2×2×1 Float32 — exact bits
- MHA Tween volume 2×2×2×1 Float32 — exact bits
- CNN1 Tween volume 2×2×2×1 Float32 — exact bits
- CNN2 Tween volume 2×2×2×1 Float32 — exact bits
- CNN3 Tween volume 2×2×2×1 Float32 — exact bits
- ConvT1 Tween volume 2×2×2×1 Float32 — exact bits
- ConvT2 Tween volume 2×2×2×1 Float32 — exact bits
- ConvT3 Tween volume 2×2×2×1 Float32 — exact bits
- RNN Tween volume 2×2×2×1 Float32 — exact bits
- LSTM Tween volume 2×2×2×1 Float32 — exact bits

#### `tweenalt_train.go` (59)

- Dense TweenAlt multi-hop Float32 — exact bits (W/loss)
- Dense TweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenAlt 1-hop Float32 — exact bits
- RMSNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenAlt 1-hop Float32 — exact bits
- LayerNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax TweenAlt 1-hop Float32 — exact bits (loss)
- Softmax TweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenAlt 1-hop Float32 — exact bits
- SwiGLU TweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential TweenAlt 1-hop Float32 — exact bits
- Sequential TweenAlt 1-hop × all 34 dtypes — spectrum
- Residual TweenAlt 1-hop Float32 — exact bits
- Residual TweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba TweenAlt 1-hop Float32 — exact bits
- Mamba TweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition TweenAlt 1-hop Float32 — exact bits
- Metacognition TweenAlt 1-hop × all 34 dtypes — spectrum
- KMeans TweenAlt 1-hop Float32 — exact bits
- KMeans TweenAlt 1-hop × all 34 dtypes — spectrum
- GDN TweenAlt 1-hop Float32 — exact bits
- NestedStack TweenAlt 1-hop Float32 — exact bits
- NestedStack TweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding TweenAlt 1-hop Float32 — exact bits
- Embedding TweenAlt 1-hop × all 34 dtypes — spectrum
- MHA TweenAlt 1-hop Float32 — exact bits
- MHA TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 TweenAlt 1-hop Float32 — exact bits
- CNN1 TweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenAlt 1-hop Float32 — exact bits
- ConvT1 TweenAlt 1-hop × all 34 dtypes — spectrum
- RNN TweenAlt 1-hop Float32 — exact bits
- RNN TweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM TweenAlt 1-hop Float32 — exact bits
- LSTM TweenAlt 1-hop × all 34 dtypes — spectrum
- Dense TweenAlt volume 1×1×1×1 Float32 — exact bits
- Dense TweenAlt volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenAlt volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenAlt volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenAlt volume 2×2×2×1 Float32 — exact bits
- Softmax TweenAlt volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenAlt volume 2×2×2×1 Float32 — exact bits
- Sequential TweenAlt volume 2×2×2×1 Float32 — exact bits
- Residual TweenAlt volume 2×2×2×1 Float32 — exact bits
- Mamba TweenAlt volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenAlt volume 2×2×2×1 Float32 — exact bits
- KMeans TweenAlt volume 2×2×2×1 Float32 — exact bits
- GDN TweenAlt volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenAlt volume 2×2×2×1 Float32 — exact bits
- Embedding TweenAlt volume 2×2×2×1 Float32 — exact bits
- MHA TweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenAlt volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenAlt volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenAlt volume 2×2×2×1 Float32 — exact bits
- RNN TweenAlt volume 2×2×2×1 Float32 — exact bits
- LSTM TweenAlt volume 2×2×2×1 Float32 — exact bits

#### `tweenchain_train.go` (59)

- Dense TweenChain multi-hop Float32 — exact bits (W/loss)
- Dense TweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenChain 1-hop Float32 — exact bits
- RMSNorm TweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenChain 1-hop Float32 — exact bits
- LayerNorm TweenChain 1-hop × all 34 dtypes — spectrum
- Softmax TweenChain 1-hop Float32 — exact bits (loss)
- Softmax TweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenChain 1-hop Float32 — exact bits
- SwiGLU TweenChain 1-hop × all 34 dtypes — spectrum
- Sequential TweenChain 1-hop Float32 — exact bits
- Sequential TweenChain 1-hop × all 34 dtypes — spectrum
- Residual TweenChain 1-hop Float32 — exact bits
- Residual TweenChain 1-hop × all 34 dtypes — spectrum
- Mamba TweenChain 1-hop Float32 — exact bits
- Mamba TweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition TweenChain 1-hop Float32 — exact bits
- Metacognition TweenChain 1-hop × all 34 dtypes — spectrum
- KMeans TweenChain 1-hop Float32 — exact bits
- KMeans TweenChain 1-hop × all 34 dtypes — spectrum
- GDN TweenChain 1-hop Float32 — exact bits
- NestedStack TweenChain 1-hop Float32 — exact bits
- NestedStack TweenChain 1-hop × all 34 dtypes — spectrum
- Embedding TweenChain 1-hop Float32 — exact bits
- Embedding TweenChain 1-hop × all 34 dtypes — spectrum
- MHA TweenChain 1-hop Float32 — exact bits
- MHA TweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 TweenChain 1-hop Float32 — exact bits
- CNN1 TweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenChain 1-hop Float32 — exact bits
- ConvT1 TweenChain 1-hop × all 34 dtypes — spectrum
- RNN TweenChain 1-hop Float32 — exact bits
- RNN TweenChain 1-hop × all 34 dtypes — spectrum
- LSTM TweenChain 1-hop Float32 — exact bits
- LSTM TweenChain 1-hop × all 34 dtypes — spectrum
- Dense TweenChain volume 1×1×1×1 Float32 — exact bits
- Dense TweenChain volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenChain volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenChain volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenChain volume 2×2×2×1 Float32 — exact bits
- Softmax TweenChain volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenChain volume 2×2×2×1 Float32 — exact bits
- Sequential TweenChain volume 2×2×2×1 Float32 — exact bits
- Residual TweenChain volume 2×2×2×1 Float32 — exact bits
- Mamba TweenChain volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenChain volume 2×2×2×1 Float32 — exact bits
- KMeans TweenChain volume 2×2×2×1 Float32 — exact bits
- GDN TweenChain volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenChain volume 2×2×2×1 Float32 — exact bits
- Embedding TweenChain volume 2×2×2×1 Float32 — exact bits
- MHA TweenChain volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenChain volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenChain volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenChain volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenChain volume 2×2×2×1 Float32 — exact bits
- RNN TweenChain volume 2×2×2×1 Float32 — exact bits
- LSTM TweenChain volume 2×2×2×1 Float32 — exact bits

#### `tweensplit_train.go` (59)

- Dense TweenSplit multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplit 1-hop Float32 — exact bits
- RMSNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplit 1-hop Float32 — exact bits
- LayerNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplit 1-hop Float32 — exact bits (loss)
- Softmax TweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplit 1-hop Float32 — exact bits
- SwiGLU TweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplit 1-hop Float32 — exact bits
- Sequential TweenSplit 1-hop × all 34 dtypes — spectrum
- Residual TweenSplit 1-hop Float32 — exact bits
- Residual TweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplit 1-hop Float32 — exact bits
- Mamba TweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplit 1-hop Float32 — exact bits
- Metacognition TweenSplit 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplit 1-hop Float32 — exact bits
- KMeans TweenSplit 1-hop × all 34 dtypes — spectrum
- GDN TweenSplit 1-hop Float32 — exact bits
- NestedStack TweenSplit 1-hop Float32 — exact bits
- NestedStack TweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplit 1-hop Float32 — exact bits
- Embedding TweenSplit 1-hop × all 34 dtypes — spectrum
- MHA TweenSplit 1-hop Float32 — exact bits
- MHA TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplit 1-hop Float32 — exact bits
- CNN1 TweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplit 1-hop Float32 — exact bits
- ConvT1 TweenSplit 1-hop × all 34 dtypes — spectrum
- RNN TweenSplit 1-hop Float32 — exact bits
- RNN TweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplit 1-hop Float32 — exact bits
- LSTM TweenSplit 1-hop × all 34 dtypes — spectrum
- Dense TweenSplit volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplit volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplit volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplit volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplit volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplit volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplit volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplit volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplit volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplit volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplit volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplit volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplit volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplit volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplit volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplit volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplit volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplit volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplit volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplit volume 2×2×2×1 Float32 — exact bits

#### `tweensplitfastproxy_train.go` (59)

- Dense TweenSplitFastProxy multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitFastProxy 1-hop Float32 — exact bits
- RMSNorm TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitFastProxy 1-hop Float32 — exact bits
- LayerNorm TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitFastProxy 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitFastProxy 1-hop Float32 — exact bits
- SwiGLU TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitFastProxy 1-hop Float32 — exact bits
- Sequential TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitFastProxy 1-hop Float32 — exact bits
- Residual TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitFastProxy 1-hop Float32 — exact bits
- Mamba TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitFastProxy 1-hop Float32 — exact bits
- Metacognition TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitFastProxy 1-hop Float32 — exact bits
- KMeans TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitFastProxy 1-hop Float32 — exact bits
- NestedStack TweenSplitFastProxy 1-hop Float32 — exact bits
- NestedStack TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitFastProxy 1-hop Float32 — exact bits
- Embedding TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitFastProxy 1-hop Float32 — exact bits
- MHA TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitFastProxy 1-hop Float32 — exact bits
- CNN1 TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitFastProxy 1-hop Float32 — exact bits
- ConvT1 TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitFastProxy 1-hop Float32 — exact bits
- RNN TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitFastProxy 1-hop Float32 — exact bits
- LSTM TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitFastProxy volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitFastProxy volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits

#### `tweensplitheadproxy_train.go` (59)

- Dense TweenSplitHeadProxy multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitHeadProxy 1-hop Float32 — exact bits
- RMSNorm TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitHeadProxy 1-hop Float32 — exact bits
- LayerNorm TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitHeadProxy 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitHeadProxy 1-hop Float32 — exact bits
- SwiGLU TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitHeadProxy 1-hop Float32 — exact bits
- Sequential TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitHeadProxy 1-hop Float32 — exact bits
- Residual TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitHeadProxy 1-hop Float32 — exact bits
- Mamba TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitHeadProxy 1-hop Float32 — exact bits
- Metacognition TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitHeadProxy 1-hop Float32 — exact bits
- KMeans TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitHeadProxy 1-hop Float32 — exact bits
- NestedStack TweenSplitHeadProxy 1-hop Float32 — exact bits
- NestedStack TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitHeadProxy 1-hop Float32 — exact bits
- Embedding TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitHeadProxy 1-hop Float32 — exact bits
- MHA TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitHeadProxy 1-hop Float32 — exact bits
- CNN1 TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitHeadProxy 1-hop Float32 — exact bits
- ConvT1 TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitHeadProxy 1-hop Float32 — exact bits
- RNN TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitHeadProxy 1-hop Float32 — exact bits
- LSTM TweenSplitHeadProxy 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitHeadProxy volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitHeadProxy volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitHeadProxy volume 2×2×2×1 Float32 — exact bits

#### `tweensplitheadproxyasync_train.go` (59)

- Dense TweenSplitHeadProxyAsync multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- RMSNorm TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- LayerNorm TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitHeadProxyAsync 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- SwiGLU TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Sequential TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Residual TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Mamba TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Metacognition TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- KMeans TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- NestedStack TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- NestedStack TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- Embedding TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- MHA TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- CNN1 TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- ConvT1 TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- RNN TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitHeadProxyAsync 1-hop Float32 — exact bits
- LSTM TweenSplitHeadProxyAsync 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitHeadProxyAsync volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitHeadProxyAsync volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitHeadProxyAsync volume 2×2×2×1 Float32 — exact bits

#### `tweensplitlinear_train.go` (59)

- Dense TweenSplitLinear multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitLinear 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitLinear 1-hop Float32 — exact bits
- RMSNorm TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitLinear 1-hop Float32 — exact bits
- LayerNorm TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitLinear 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitLinear 1-hop Float32 — exact bits
- SwiGLU TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitLinear 1-hop Float32 — exact bits
- Sequential TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitLinear 1-hop Float32 — exact bits
- Residual TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitLinear 1-hop Float32 — exact bits
- Mamba TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitLinear 1-hop Float32 — exact bits
- Metacognition TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitLinear 1-hop Float32 — exact bits
- KMeans TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitLinear 1-hop Float32 — exact bits
- NestedStack TweenSplitLinear 1-hop Float32 — exact bits
- NestedStack TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitLinear 1-hop Float32 — exact bits
- Embedding TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitLinear 1-hop Float32 — exact bits
- MHA TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitLinear 1-hop Float32 — exact bits
- CNN1 TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitLinear 1-hop Float32 — exact bits
- ConvT1 TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitLinear 1-hop Float32 — exact bits
- RNN TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitLinear 1-hop Float32 — exact bits
- LSTM TweenSplitLinear 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitLinear volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitLinear volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitLinear volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitLinear volume 2×2×2×1 Float32 — exact bits

#### `tweensplitlinearcache_train.go` (59)

- Dense TweenSplitLinearCache multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitLinearCache 1-hop Float32 — exact bits
- RMSNorm TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitLinearCache 1-hop Float32 — exact bits
- LayerNorm TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitLinearCache 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitLinearCache 1-hop Float32 — exact bits
- SwiGLU TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitLinearCache 1-hop Float32 — exact bits
- Sequential TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitLinearCache 1-hop Float32 — exact bits
- Residual TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitLinearCache 1-hop Float32 — exact bits
- Mamba TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitLinearCache 1-hop Float32 — exact bits
- Metacognition TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitLinearCache 1-hop Float32 — exact bits
- KMeans TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitLinearCache 1-hop Float32 — exact bits
- NestedStack TweenSplitLinearCache 1-hop Float32 — exact bits
- NestedStack TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitLinearCache 1-hop Float32 — exact bits
- Embedding TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitLinearCache 1-hop Float32 — exact bits
- MHA TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitLinearCache 1-hop Float32 — exact bits
- CNN1 TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitLinearCache 1-hop Float32 — exact bits
- ConvT1 TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitLinearCache 1-hop Float32 — exact bits
- RNN TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitLinearCache 1-hop Float32 — exact bits
- LSTM TweenSplitLinearCache 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitLinearCache volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitLinearCache volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitLinearCache volume 2×2×2×1 Float32 — exact bits

#### `tweensplitsparse_train.go` (59)

- Dense TweenSplitSparse multi-hop Float32 — exact bits (W/loss)
- Dense TweenSplitSparse 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplitSparse 1-hop Float32 — exact bits
- RMSNorm TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplitSparse 1-hop Float32 — exact bits
- LayerNorm TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplitSparse 1-hop Float32 — exact bits (loss)
- Softmax TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplitSparse 1-hop Float32 — exact bits
- SwiGLU TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplitSparse 1-hop Float32 — exact bits
- Sequential TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Residual TweenSplitSparse 1-hop Float32 — exact bits
- Residual TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplitSparse 1-hop Float32 — exact bits
- Mamba TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplitSparse 1-hop Float32 — exact bits
- Metacognition TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- KMeans TweenSplitSparse 1-hop Float32 — exact bits
- KMeans TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- GDN TweenSplitSparse 1-hop Float32 — exact bits
- NestedStack TweenSplitSparse 1-hop Float32 — exact bits
- NestedStack TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplitSparse 1-hop Float32 — exact bits
- Embedding TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- MHA TweenSplitSparse 1-hop Float32 — exact bits
- MHA TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplitSparse 1-hop Float32 — exact bits
- CNN1 TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplitSparse 1-hop Float32 — exact bits
- ConvT1 TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- RNN TweenSplitSparse 1-hop Float32 — exact bits
- RNN TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplitSparse 1-hop Float32 — exact bits
- LSTM TweenSplitSparse 1-hop × all 34 dtypes — spectrum
- Dense TweenSplitSparse volume 1×1×1×1 Float32 — exact bits
- Dense TweenSplitSparse volume 2×1×1×1 chain Float32 — exact bits
- Dense TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RMSNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LayerNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Softmax TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- SwiGLU TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Sequential TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Residual TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Mamba TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Metacognition TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- KMeans TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- GDN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- NestedStack TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- Embedding TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- MHA TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- CNN3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- ConvT3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- RNN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits
- LSTM TweenSplitSparse volume 2×2×2×1 Float32 — exact bits

### suites/go_v_cuda (939 leaf cases)

#### `cnn.go` (8)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- CNN1 Float32 Linear pad — exact bits vs Go
- CNN2 Float32 ReLU — exact bits vs Go
- CNN3 Float32 Linear — exact bits vs Go
- CNN1 FormatNone ×34 — exact_bit + Go/C++/CUDA timings
- CNN2 FormatNone ×34 — exact_bit + Go/C++/CUDA timings
- CNN3 FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `convt.go` (8)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- ConvT1 Float32 Linear pad — exact bits vs Go
- ConvT2 Float32 ReLU — exact bits vs Go
- ConvT3 Float32 Linear out_pad — exact bits vs Go
- ConvT1 FormatNone ×34 — exact_bit + Go/C++/CUDA timings
- ConvT2 FormatNone ×34 — exact_bit + Go/C++/CUDA timings
- ConvT3 FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `dense_train.go` (7)

- Dense NormalBP leaf Float32 ReLU — exact tensors; CUDA loss band (dX/dW/W/loss)
- Dense NormalBP leaf × all 34 dtypes — spectrum + speed
- Dense NormalBP volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense NormalBP volume 1×1×1×2 chain Float32 — exact tensors; CUDA loss band
- Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Dense NormalBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `embedding.go` (6)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Embedding Float32 Linear [B,T] — exact bits vs Go
- Embedding Float32 Linear [T] rank1 — exact bits vs Go
- Embedding Float32 Linear [B,T,1] rank3 — exact bits vs Go
- Embedding FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `embedding_train.go` (7)

- Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Embedding NormalBP leaf × all 34 dtypes — spectrum + speed
- Embedding NormalBP volume 1×1×1×1 Float32 — exact bits
- Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Embedding NormalBP volume 2×2×2×1 Float32 — exact bits
- Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `gdn.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- GDN Float32 primary — ≥industry vs Go, exact vs C++
- GDN Float32 alt GQA — ≥industry vs Go, exact vs C++
- GDN Float32 mid — parity + Go/C++/CUDA timings

#### `gdn_train.go` (4)

- GDN NormalBP leaf Float32 — ≥loose vs Go (+ speed)
- GDN NormalBP volume 1×1×1×1 Float32 — ≥industry vs Go
- GDN NormalBP volume 1×1×1×2 chain Float32 — ≥industry vs Go
- GDN NormalBP volume 2×2×2×1 Float32 — ≥industry vs Go

#### `kmeans.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- KMeans probabilities Float32 Linear — exact bits vs Go
- KMeans features Float32 Linear — exact bits vs Go
- KMeans both modes ×34 — exact_bit + Go/C++/CUDA timings

#### `kmeans_train.go` (8)

- KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed
- KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits
- KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `layernorm.go` (6)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits vs Go
- LayerNorm Float32 [batch,seq,dim] random affine — exact bits vs Go
- LayerNorm Float32 custom eps — exact bits vs Go
- LayerNorm FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `layernorm_train.go` (7)

- LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)
- LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `lstm.go` (4)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- LSTM Float32 — exact bits vs Go
- LSTM FormatNone ×34 — ≥98% + Go/C++/CUDA timings

#### `lstm_train.go` (7)

- LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- LSTM NormalBP leaf × all 34 dtypes — spectrum + speed
- LSTM NormalBP volume 1×1×1×1 Float32 — exact bits
- LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LSTM NormalBP volume 2×2×2×1 Float32 — exact bits
- LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mamba.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Mamba Float32 [B,T,D] — exact bits vs Go
- Mamba Float32 alt shape — exact bits vs Go
- Mamba FormatNone ×34 — ≥99% + Go/C++/CUDA timings

#### `mamba_train.go` (7)

- Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Mamba NormalBP leaf × all 34 dtypes — spectrum + speed
- Mamba NormalBP volume 1×1×1×1 Float32 — exact bits
- Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Mamba NormalBP volume 2×2×2×1 Float32 — exact bits
- Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `meshtween_train.go` (28)

- Dense MeshTween mesh-clock 111 × all 34 dtypes
- Dense MeshTween mesh-clock 211 × all 34 dtypes
- Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `meshtweenchain_train.go` (28)

- Dense MeshTweenChain mesh-clock 111 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 211 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `metacognition.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Metacognition Float32 [B,D] — exact bits vs Go
- Metacognition Float32 [B,T,D] — exact bits vs Go
- Metacognition ×34 — ≥99% + Go/C++/CUDA timings

#### `metacognition_train.go` (7)

- Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed
- Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits
- Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mha.go` (6)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- MHA DecoderCausal Float32 — exact bits vs Go
- MHA GQA 4h/2kv Float32 — exact bits vs Go
- MHA Encoder bidirectional Float32 — exact bits vs Go
- MHA DecoderCausal FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `mha_train.go` (8)

- MHA NormalBP leaf DecoderCausal Float32 — exact bits
- MHA NormalBP leaf GQA + Encoder Float32 — exact bits
- MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed
- MHA NormalBP volume 1×1×1×1 Float32 — exact bits
- MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits
- MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- MHA NormalBP volume 2×2×2×1 Float32 — exact bits
- MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `parallel.go` (11)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Parallel Add Float32 [B,D] — exact bits vs Go
- Parallel Add Float32 [B,T,D] — exact bits vs Go
- Parallel Concat Float32 — exact bits vs Go
- Parallel Avg Float32 — exact bits vs Go
- Parallel Filter Float32 — exact bits vs Go
- Parallel Max Float32 — exact bits vs Go
- Parallel SparseK Float32 — exact bits vs Go
- Parallel Disagree Float32 (2+3 branches) — exact bits vs Go
- Parallel Add ×34 — ≥99% + Go/C++/CUDA timings

#### `parallel_nbp_train.go` (9)

- Dense Add Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP × all 34 dtypes — spectrum
- Dense Avg Parallel NormalBP Float32 — exact bits
- Dense Concat Parallel NormalBP Float32 — exact bits
- Dense Max Parallel NormalBP Float32 — exact bits
- Dense SparseK Parallel NormalBP Float32 — exact bits
- Dense Disagree Parallel NormalBP Float32 — exact bits
- Dense Filter Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits

#### `residual.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Residual Dense Float32 [B,D] — exact bits vs Go
- Residual Dense Float32 [B,T,D] — exact bits vs Go
- Residual Dense FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `residual_train.go` (7)

- Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Residual NormalBP leaf × all 34 dtypes — spectrum + speed
- Residual NormalBP volume 1×1×1×1 Float32 — exact bits
- Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Residual NormalBP volume 2×2×2×1 Float32 — exact bits
- Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rmsnorm.go` (6)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- RMSNorm Float32 [batch,dim] γ=1 — exact bits vs Go
- RMSNorm Float32 [batch,seq,dim] random γ — exact bits vs Go
- RMSNorm Float32 custom eps — exact bits vs Go
- RMSNorm FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `rmsnorm_train.go` (7)

- RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)
- RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rnn.go` (4)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- RNN Float32 — exact bits vs Go
- RNN FormatNone ×34 — ≥99% + Go/C++/CUDA timings

#### `rnn_train.go` (7)

- RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- RNN NormalBP leaf × all 34 dtypes — spectrum + speed
- RNN NormalBP volume 1×1×1×1 Float32 — exact bits
- RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RNN NormalBP volume 2×2×2×1 Float32 — exact bits
- RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `seqmix.go` (4)

- Locate welvet.cuda tool
- seqmix Kind iota + String — Go
- seqmix mixer tags — Go MHA/Mamba/GDN
- seqmix Kind names + tags — Go vs CUDA WSX1

#### `sequential.go` (5)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Sequential Dense Float32 [B,D] — exact bits vs Go
- Sequential Dense Float32 [B,T,D] — exact bits vs Go
- Sequential Dense FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `sequential_train.go` (7)

- Sequential NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Sequential NormalBP leaf × all 34 dtypes — spectrum + speed
- Sequential NormalBP volume 1×1×1×1 Float32 — exact bits
- Sequential NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Sequential NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Sequential NormalBP volume 2×2×2×1 Float32 — exact bits
- Sequential NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `softmax.go` (11)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- Softmax Standard Float32 — exact bits vs Go
- Softmax Temperature Float32 — exact bits vs Go
- Softmax Masked Float32 — exact bits vs Go
- Softmax Sparse Float32 — exact bits vs Go
- Softmax Entmax Float32 — exact bits vs Go
- Softmax Grid Float32 — exact bits vs Go
- Softmax Hierarchical Float32 — exact bits vs Go
- Softmax Gumbel Float32 — smoke (non-deterministic)
- Softmax all Kind ×34 — exact_bit + Go/C++/CUDA timings

#### `softmax_train.go` (9)

- Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)
- Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)
- Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed
- Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum
- Softmax NormalBP volume 1×1×1×1 Float32 — exact bits
- Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Softmax NormalBP volume 2×2×2×1 Float32 — exact bits
- Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stack_train.go` (7)

- Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Stack NormalBP leaf × all 34 dtypes — spectrum + speed
- Stack NormalBP volume 1×1×1×1 Float32 — exact bits
- Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Stack NormalBP volume 2×2×2×1 Float32 — exact bits
- Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stepbp_train.go` (84)

- Dense StepBP multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense StepBP 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepBP 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm StepBP 1-hop × all 34 dtypes — spectrum
- LayerNorm StepBP 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm StepBP 1-hop × all 34 dtypes — spectrum
- Softmax StepBP 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax StepBP 1-hop × all 34 dtypes — spectrum
- SwiGLU StepBP 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU StepBP 1-hop × all 34 dtypes — spectrum
- Sequential StepBP 1-hop Float32 — exact tensors; CUDA loss band
- Sequential StepBP 1-hop × all 34 dtypes — spectrum
- Residual StepBP 1-hop Float32 — exact tensors; CUDA loss band
- Residual StepBP 1-hop × all 34 dtypes — spectrum
- Mamba StepBP 1-hop Float32 — exact tensors; CUDA loss band
- Mamba StepBP 1-hop × all 34 dtypes — spectrum
- Metacognition StepBP 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition StepBP 1-hop × all 34 dtypes — spectrum
- CNN2 StepBP rank 4 × all 34 dtypes
- CNN3 StepBP rank 5 × all 34 dtypes
- ConvT2 StepBP rank 4 × all 34 dtypes
- ConvT3 StepBP rank 5 × all 34 dtypes
- Dense ModeMeshBP volume 211 × all 34 dtypes
- RMSNorm ModeMeshBP volume 211 × all 34 dtypes
- LayerNorm ModeMeshBP volume 211 × all 34 dtypes
- SwiGLU ModeMeshBP volume 211 × all 34 dtypes
- Softmax ModeMeshBP volume 211 × all 34 dtypes
- Embedding ModeMeshBP volume 211 × all 34 dtypes
- MHA ModeMeshBP volume 211 × all 34 dtypes
- RNN ModeMeshBP volume 211 × all 34 dtypes
- LSTM ModeMeshBP volume 211 × all 34 dtypes
- Sequential ModeMeshBP volume 211 × all 34 dtypes
- Residual ModeMeshBP volume 211 × all 34 dtypes
- Mamba ModeMeshBP volume 211 × all 34 dtypes
- Metacognition ModeMeshBP volume 211 × all 34 dtypes
- NestedStack ModeMeshBP volume 211 × all 34 dtypes
- CNN1 ModeMeshBP volume 211 × all 34 dtypes
- CNN2 ModeMeshBP volume 211 × all 34 dtypes
- CNN3 ModeMeshBP volume 211 × all 34 dtypes
- ConvT1 ModeMeshBP volume 211 × all 34 dtypes
- ConvT2 ModeMeshBP volume 211 × all 34 dtypes
- ConvT3 ModeMeshBP volume 211 × all 34 dtypes
- Dense MeshBP mesh-clock 111 × all 34 dtypes
- Dense MeshBP mesh-clock 222 × all 34 dtypes
- Dense MeshBP disabled cell and remote link
- GDN StepBP 1-hop Float32 — ≥strict vs Go (scan FP)
- NestedStack StepBP 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepBP 1-hop × all 34 dtypes — spectrum
- Embedding StepBP 1-hop Float32 — exact tensors; CUDA loss band
- Embedding StepBP 1-hop × all 34 dtypes — spectrum
- MHA StepBP 1-hop Float32 — exact tensors; CUDA loss band
- MHA StepBP 1-hop × all 34 dtypes — spectrum
- CNN1 StepBP 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 StepBP 1-hop × all 34 dtypes — spectrum
- ConvT1 StepBP 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 StepBP 1-hop × all 34 dtypes — spectrum
- RNN StepBP 1-hop Float32 — exact tensors; CUDA loss band
- RNN StepBP 1-hop × all 34 dtypes — spectrum
- LSTM StepBP 1-hop Float32 — exact tensors; CUDA loss band
- LSTM StepBP 1-hop × all 34 dtypes — spectrum
- Dense MeshBP 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshBP 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshBP 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshBP volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `steptween_train.go` (60)

- Dense StepTween multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense StepTween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTween 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm StepTween 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTween 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm StepTween 1-hop × all 34 dtypes — spectrum
- Softmax StepTween 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax StepTween 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTween 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU StepTween 1-hop × all 34 dtypes — spectrum
- Sequential StepTween 1-hop Float32 — exact tensors; CUDA loss band
- Sequential StepTween 1-hop × all 34 dtypes — spectrum
- Residual StepTween 1-hop Float32 — exact tensors; CUDA loss band
- Residual StepTween 1-hop × all 34 dtypes — spectrum
- Mamba StepTween 1-hop Float32 — exact tensors; CUDA loss band
- Mamba StepTween 1-hop × all 34 dtypes — spectrum
- Embedding StepTween 1-hop Float32 — exact tensors; CUDA loss band
- Embedding StepTween 1-hop × all 34 dtypes — spectrum
- MHA StepTween 1-hop Float32 — exact tensors; CUDA loss band
- MHA StepTween 1-hop × all 34 dtypes — spectrum
- CNN1 StepTween 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 StepTween 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTween 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 StepTween 1-hop × all 34 dtypes — spectrum
- RNN StepTween 1-hop Float32 — exact tensors; CUDA loss band
- RNN StepTween 1-hop × all 34 dtypes — spectrum
- LSTM StepTween 1-hop Float32 — exact tensors; CUDA loss band
- LSTM StepTween 1-hop × all 34 dtypes — spectrum
- Metacognition StepTween Float32
- Metacognition StepTween × all 34 dtypes
- GDN StepTween Float32 — ≥strict weights; CUDA loss band
- NestedStack StepTween Float32
- NestedStack StepTween × all 34 dtypes
- Metacognition StepTween volume 222 Float32
- GDN StepTween volume 222 Float32
- NestedStack StepTween volume 222 Float32
- CNN2 StepTween rank 4 × all 34 dtypes
- CNN3 StepTween rank 5 × all 34 dtypes
- ConvT2 StepTween rank 4 × all 34 dtypes
- ConvT3 StepTween rank 5 × all 34 dtypes
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `steptweenalt_train.go` (60)

- Dense StepTweenAlt multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense StepTweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax StepTweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Sequential StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Residual StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Residual StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Mamba StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenAlt rank 4 × all 34 dtypes
- CNN3 StepTweenAlt rank 5 × all 34 dtypes
- ConvT2 StepTweenAlt rank 4 × all 34 dtypes
- ConvT3 StepTweenAlt rank 5 × all 34 dtypes
- GDN StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Embedding StepTweenAlt 1-hop × all 34 dtypes — spectrum
- MHA StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- MHA StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- RNN StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- RNN StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- LSTM StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `steptweenchain_train.go` (60)

- Dense StepTweenChain multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum
- Residual StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Residual StepTweenChain 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenChain rank 4 × all 34 dtypes
- CNN3 StepTweenChain rank 5 × all 34 dtypes
- ConvT2 StepTweenChain rank 4 × all 34 dtypes
- ConvT3 StepTweenChain rank 5 × all 34 dtypes
- GDN StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum
- MHA StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- MHA StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- RNN StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- RNN StepTweenChain 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenChain 1-hop Float32 — exact tensors; CUDA loss band
- LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `steptweensplit_train.go` (60)

- Dense StepTweenSplit multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense StepTweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax StepTweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Sequential StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Residual StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Mamba StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenSplit rank 4 × all 34 dtypes
- CNN3 StepTweenSplit rank 5 × all 34 dtypes
- ConvT2 StepTweenSplit rank 4 × all 34 dtypes
- ConvT3 StepTweenSplit rank 5 × all 34 dtypes
- GDN StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Embedding StepTweenSplit 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- MHA StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- RNN StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- LSTM StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `suite.go` (7)

- Locate welvet.cuda welvet_dense_fwd
- Locate welvet.cpp welvet_dense_fwd (optional speed)
- CUDA device identity
- Spectrum ladder self-check
- Dense forward identity+bias Float32 CUDA — exact bits vs Go
- Dense forward ReLU Float32 CUDA — exact bits vs Go
- CUDA FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `swiglu.go` (6)

- Locate welvet.cuda tool
- Locate welvet.cpp tool (optional speed)
- SwiGLU Float32 [batch,in] — exact bits vs Go
- SwiGLU Float32 [batch,seq,in] — exact bits vs Go
- SwiGLU DefaultFFN Float32 — exact bits vs Go
- SwiGLU FormatNone ×34 — exact_bit + Go/C++/CUDA timings

#### `swiglu_train.go` (7)

- SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed
- SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `tween_train.go` (80)

- Dense Tween multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense Tween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Tween 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm Tween 1-hop × all 34 dtypes — spectrum
- LayerNorm Tween 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm Tween 1-hop × all 34 dtypes — spectrum
- Softmax Tween 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax Tween 1-hop × all 34 dtypes — spectrum
- SwiGLU Tween 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU Tween 1-hop × all 34 dtypes — spectrum
- Sequential Tween 1-hop Float32 — exact tensors; CUDA loss band
- Sequential Tween 1-hop × all 34 dtypes — spectrum
- Residual Tween 1-hop Float32 — exact tensors; CUDA loss band
- Residual Tween 1-hop × all 34 dtypes — spectrum
- Mamba Tween 1-hop Float32 — exact tensors; CUDA loss band
- Mamba Tween 1-hop × all 34 dtypes — spectrum
- Metacognition Tween 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition Tween 1-hop × all 34 dtypes — spectrum
- CNN2 Tween rank 4 × all 34 dtypes
- CNN3 Tween rank 5 × all 34 dtypes
- ConvT2 Tween rank 4 × all 34 dtypes
- ConvT3 Tween rank 5 × all 34 dtypes
- GDN Tween 1-hop Float32 — ≥strict weights; CUDA loss band
- NestedStack Tween 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack Tween 1-hop × all 34 dtypes — spectrum
- Embedding Tween 1-hop Float32 — exact tensors; CUDA loss band
- Embedding Tween 1-hop × all 34 dtypes — spectrum
- MHA Tween 1-hop Float32 — exact tensors; CUDA loss band
- MHA Tween 1-hop × all 34 dtypes — spectrum
- CNN1 Tween 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 Tween 1-hop × all 34 dtypes — spectrum
- ConvT1 Tween 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 Tween 1-hop × all 34 dtypes — spectrum
- RNN Tween 1-hop Float32 — exact tensors; CUDA loss band
- RNN Tween 1-hop × all 34 dtypes — spectrum
- LSTM Tween 1-hop Float32 — exact tensors; CUDA loss band
- LSTM Tween 1-hop × all 34 dtypes — spectrum
- RMSNorm Tween volume 211 × all 34 dtypes
- LayerNorm Tween volume 211 × all 34 dtypes
- SwiGLU Tween volume 211 × all 34 dtypes
- Softmax Tween volume 211 × all 34 dtypes
- Embedding Tween volume 211 × all 34 dtypes
- MHA Tween volume 211 × all 34 dtypes
- RNN Tween volume 211 × all 34 dtypes
- LSTM Tween volume 211 × all 34 dtypes
- Seq Tween volume 211 × all 34 dtypes
- Re Tween volume 211 × all 34 dtypes
- Mb Tween volume 211 × all 34 dtypes
- Mc Tween volume 211 × all 34 dtypes
- Sk Tween volume 211 × all 34 dtypes
- CNN1 Tween volume 211 × all 34 dtypes
- CNN2 Tween volume 211 × all 34 dtypes
- CNN3 Tween volume 211 × all 34 dtypes
- ConvT1 Tween volume 211 × all 34 dtypes
- ConvT2 Tween volume 211 × all 34 dtypes
- ConvT3 Tween volume 211 × all 34 dtypes
- Dense Tween volume 211 × all 34 dtypes
- Dense Tween volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense Tween volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM Tween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `tweenalt_train.go` (80)

- Dense TweenAlt multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense TweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax TweenAlt 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax TweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU TweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Sequential TweenAlt 1-hop × all 34 dtypes — spectrum
- Residual TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Residual TweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Mamba TweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 TweenAlt rank 4 × all 34 dtypes
- CNN3 TweenAlt rank 5 × all 34 dtypes
- ConvT2 TweenAlt rank 4 × all 34 dtypes
- ConvT3 TweenAlt rank 5 × all 34 dtypes
- GDN TweenAlt 1-hop Float32 — ≥strict weights; finite GDN loss band
- NestedStack TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack TweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- Embedding TweenAlt 1-hop × all 34 dtypes — spectrum
- MHA TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- MHA TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 TweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 TweenAlt 1-hop × all 34 dtypes — spectrum
- RNN TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- RNN TweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM TweenAlt 1-hop Float32 — exact tensors; CUDA loss band
- LSTM TweenAlt 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenAlt volume 211 × all 34 dtypes
- LayerNorm TweenAlt volume 211 × all 34 dtypes
- SwiGLU TweenAlt volume 211 × all 34 dtypes
- Softmax TweenAlt volume 211 × all 34 dtypes
- Embedding TweenAlt volume 211 × all 34 dtypes
- MHA TweenAlt volume 211 × all 34 dtypes
- RNN TweenAlt volume 211 × all 34 dtypes
- LSTM TweenAlt volume 211 × all 34 dtypes
- Seq TweenAlt volume 211 × all 34 dtypes
- Re TweenAlt volume 211 × all 34 dtypes
- Mb TweenAlt volume 211 × all 34 dtypes
- Mc TweenAlt volume 211 × all 34 dtypes
- Sk TweenAlt volume 211 × all 34 dtypes
- CNN1 TweenAlt volume 211 × all 34 dtypes
- CNN2 TweenAlt volume 211 × all 34 dtypes
- CNN3 TweenAlt volume 211 × all 34 dtypes
- ConvT1 TweenAlt volume 211 × all 34 dtypes
- ConvT2 TweenAlt volume 211 × all 34 dtypes
- ConvT3 TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense TweenAlt volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM TweenAlt volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `tweenchain_train.go` (80)

- Dense TweenChain multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense TweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm TweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm TweenChain 1-hop × all 34 dtypes — spectrum
- Softmax TweenChain 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax TweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU TweenChain 1-hop × all 34 dtypes — spectrum
- Sequential TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Sequential TweenChain 1-hop × all 34 dtypes — spectrum
- Residual TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Residual TweenChain 1-hop × all 34 dtypes — spectrum
- Mamba TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Mamba TweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition TweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 TweenChain rank 4 × all 34 dtypes
- CNN3 TweenChain rank 5 × all 34 dtypes
- ConvT2 TweenChain rank 4 × all 34 dtypes
- ConvT3 TweenChain rank 5 × all 34 dtypes
- GDN TweenChain 1-hop Float32 — ≥strict weights; CUDA loss band
- NestedStack TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack TweenChain 1-hop × all 34 dtypes — spectrum
- Embedding TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- Embedding TweenChain 1-hop × all 34 dtypes — spectrum
- MHA TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- MHA TweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 TweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 TweenChain 1-hop × all 34 dtypes — spectrum
- RNN TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- RNN TweenChain 1-hop × all 34 dtypes — spectrum
- LSTM TweenChain 1-hop Float32 — exact tensors; CUDA loss band
- LSTM TweenChain 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenChain volume 211 × all 34 dtypes
- LayerNorm TweenChain volume 211 × all 34 dtypes
- SwiGLU TweenChain volume 211 × all 34 dtypes
- Softmax TweenChain volume 211 × all 34 dtypes
- Embedding TweenChain volume 211 × all 34 dtypes
- MHA TweenChain volume 211 × all 34 dtypes
- RNN TweenChain volume 211 × all 34 dtypes
- LSTM TweenChain volume 211 × all 34 dtypes
- Seq TweenChain volume 211 × all 34 dtypes
- Re TweenChain volume 211 × all 34 dtypes
- Mb TweenChain volume 211 × all 34 dtypes
- Mc TweenChain volume 211 × all 34 dtypes
- Sk TweenChain volume 211 × all 34 dtypes
- CNN1 TweenChain volume 211 × all 34 dtypes
- CNN2 TweenChain volume 211 × all 34 dtypes
- CNN3 TweenChain volume 211 × all 34 dtypes
- ConvT1 TweenChain volume 211 × all 34 dtypes
- ConvT2 TweenChain volume 211 × all 34 dtypes
- ConvT3 TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense TweenChain volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM TweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

#### `tweensplit_train.go` (80)

- Dense TweenSplit multi-hop Float32 — exact tensors; CUDA loss band (W/loss)
- Dense TweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- RMSNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- LayerNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplit 1-hop Float32 — exact tensors; CUDA loss band (loss)
- Softmax TweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- SwiGLU TweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Sequential TweenSplit 1-hop × all 34 dtypes — spectrum
- Residual TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Residual TweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Mamba TweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Metacognition TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 TweenSplit rank 4 × all 34 dtypes
- CNN3 TweenSplit rank 5 × all 34 dtypes
- ConvT2 TweenSplit rank 4 × all 34 dtypes
- ConvT3 TweenSplit rank 5 × all 34 dtypes
- GDN TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- NestedStack TweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- Embedding TweenSplit 1-hop × all 34 dtypes — spectrum
- MHA TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- MHA TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- CNN1 TweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- ConvT1 TweenSplit 1-hop × all 34 dtypes — spectrum
- RNN TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- RNN TweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplit 1-hop Float32 — exact tensors; CUDA loss band
- LSTM TweenSplit 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenSplit volume 211 × all 34 dtypes
- LayerNorm TweenSplit volume 211 × all 34 dtypes
- SwiGLU TweenSplit volume 211 × all 34 dtypes
- Softmax TweenSplit volume 211 × all 34 dtypes
- Embedding TweenSplit volume 211 × all 34 dtypes
- MHA TweenSplit volume 211 × all 34 dtypes
- RNN TweenSplit volume 211 × all 34 dtypes
- LSTM TweenSplit volume 211 × all 34 dtypes
- Seq TweenSplit volume 211 × all 34 dtypes
- Re TweenSplit volume 211 × all 34 dtypes
- Mb TweenSplit volume 211 × all 34 dtypes
- Mc TweenSplit volume 211 × all 34 dtypes
- Sk TweenSplit volume 211 × all 34 dtypes
- CNN1 TweenSplit volume 211 × all 34 dtypes
- CNN2 TweenSplit volume 211 × all 34 dtypes
- CNN3 TweenSplit volume 211 × all 34 dtypes
- ConvT1 TweenSplit volume 211 × all 34 dtypes
- ConvT2 TweenSplit volume 211 × all 34 dtypes
- ConvT3 TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 1×1×1×1 Float32 — exact tensors; CUDA loss band
- Dense TweenSplit volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band
- Dense TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RMSNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LayerNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Softmax TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- SwiGLU TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Sequential TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Residual TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Mamba TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Metacognition TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- GDN TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- NestedStack TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- Embedding TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- MHA TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- CNN3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- ConvT3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- RNN TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band
- LSTM TweenSplit volume 2×2×2×1 Float32 — exact tensors; CUDA loss band

### suites/go_v_metal (961 leaf cases)

#### `cnn.go` (8)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- CNN1 Float32 Linear pad — exact bits vs Go
- CNN2 Float32 ReLU — exact bits vs Go
- CNN3 Float32 Linear — exact bits vs Go
- CNN1 FormatNone ×34 — exact_bit + Go/C++/Metal timings
- CNN2 FormatNone ×34 — exact_bit + Go/C++/Metal timings
- CNN3 FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `convt.go` (8)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- ConvT1 Float32 Linear pad — exact bits vs Go
- ConvT2 Float32 ReLU — exact bits vs Go
- ConvT3 Float32 Linear out_pad — exact bits vs Go
- ConvT1 FormatNone ×34 — exact_bit + Go/C++/Metal timings
- ConvT2 FormatNone ×34 — exact_bit + Go/C++/Metal timings
- ConvT3 FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `dense_train.go` (7)

- Dense NormalBP leaf Float32 ReLU — exact tensors; Metal loss band (dX/dW/W/loss)
- Dense NormalBP leaf × all 34 dtypes — spectrum + speed
- Dense NormalBP volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense NormalBP volume 1×1×1×2 chain Float32 — exact tensors; Metal loss band
- Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Dense NormalBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `embedding.go` (6)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Embedding Float32 Linear [B,T] — exact bits vs Go
- Embedding Float32 Linear [T] rank1 — exact bits vs Go
- Embedding Float32 Linear [B,T,1] rank3 — exact bits vs Go
- Embedding FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `embedding_train.go` (7)

- Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Embedding NormalBP leaf × all 34 dtypes — spectrum + speed
- Embedding NormalBP volume 1×1×1×1 Float32 — exact bits
- Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Embedding NormalBP volume 2×2×2×1 Float32 — exact bits
- Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `gdn.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- GDN Float32 primary — ≥industry vs Go, exact vs C++
- GDN Float32 alt GQA — ≥industry vs Go, exact vs C++
- GDN Float32 mid — parity + Go/C++/Metal timings

#### `gdn_train.go` (4)

- GDN NormalBP leaf Float32 — ≥loose vs Go (+ speed)
- GDN NormalBP volume 1×1×1×1 Float32 — ≥industry vs Go
- GDN NormalBP volume 1×1×1×2 chain Float32 — ≥industry vs Go
- GDN NormalBP volume 2×2×2×1 Float32 — ≥industry vs Go

#### `kmeans.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- KMeans probabilities Float32 Linear — exact bits vs Go
- KMeans features Float32 Linear — exact bits vs Go
- KMeans both modes ×34 — exact_bit + Go/C++/Metal timings

#### `kmeans_train.go` (8)

- KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed
- KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits
- KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `layernorm.go` (6)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits vs Go
- LayerNorm Float32 [batch,seq,dim] random affine — exact bits vs Go
- LayerNorm Float32 custom eps — exact bits vs Go
- LayerNorm FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `layernorm_train.go` (7)

- LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)
- LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `lstm.go` (4)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- LSTM Float32 — exact bits vs Go
- LSTM FormatNone ×34 — ≥98% + Go/C++/Metal timings

#### `lstm_train.go` (7)

- LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- LSTM NormalBP leaf × all 34 dtypes — spectrum + speed
- LSTM NormalBP volume 1×1×1×1 Float32 — exact bits
- LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LSTM NormalBP volume 2×2×2×1 Float32 — exact bits
- LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mamba.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Mamba Float32 [B,T,D] — exact bits vs Go
- Mamba Float32 alt shape — exact bits vs Go
- Mamba FormatNone ×34 — ≥99% + Go/C++/Metal timings

#### `mamba_train.go` (7)

- Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Mamba NormalBP leaf × all 34 dtypes — spectrum + speed
- Mamba NormalBP volume 1×1×1×1 Float32 — exact bits
- Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Mamba NormalBP volume 2×2×2×1 Float32 — exact bits
- Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `meshtween_train.go` (28)

- Dense MeshTween mesh-clock 111 × all 34 dtypes
- Dense MeshTween mesh-clock 211 × all 34 dtypes
- Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact tensors; Metal loss band
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `meshtweenchain_train.go` (28)

- Dense MeshTweenChain mesh-clock 111 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 211 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `metacognition.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Metacognition Float32 [B,D] — exact bits vs Go
- Metacognition Float32 [B,T,D] — exact bits vs Go
- Metacognition ×34 — ≥99% + Go/C++/Metal timings

#### `metacognition_train.go` (7)

- Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed
- Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits
- Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mha.go` (6)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- MHA DecoderCausal Float32 — exact bits vs Go
- MHA GQA 4h/2kv Float32 — exact bits vs Go
- MHA Encoder bidirectional Float32 — exact bits vs Go
- MHA DecoderCausal FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `mha_train.go` (8)

- MHA NormalBP leaf DecoderCausal Float32 — exact bits
- MHA NormalBP leaf GQA + Encoder Float32 — exact bits
- MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed
- MHA NormalBP volume 1×1×1×1 Float32 — exact bits
- MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits
- MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- MHA NormalBP volume 2×2×2×1 Float32 — exact bits
- MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `parallel.go` (11)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Parallel Add Float32 [B,D] — exact bits vs Go
- Parallel Add Float32 [B,T,D] — exact bits vs Go
- Parallel Concat Float32 — exact bits vs Go
- Parallel Avg Float32 — exact bits vs Go
- Parallel Filter Float32 — exact bits vs Go
- Parallel Max Float32 — exact bits vs Go
- Parallel SparseK Float32 — exact bits vs Go
- Parallel Disagree Float32 (2+3 branches) — exact bits vs Go
- Parallel Add ×34 — ≥99% + Go/C++/Metal timings

#### `parallel_nbp_train.go` (9)

- Dense Add Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP × all 34 dtypes — spectrum
- Dense Avg Parallel NormalBP Float32 — exact bits
- Dense Concat Parallel NormalBP Float32 — exact bits
- Dense Max Parallel NormalBP Float32 — exact bits
- Dense SparseK Parallel NormalBP Float32 — exact bits
- Dense Disagree Parallel NormalBP Float32 — exact bits
- Dense Filter Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits

#### `residual.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Residual Dense Float32 [B,D] — exact bits vs Go
- Residual Dense Float32 [B,T,D] — exact bits vs Go
- Residual Dense FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `residual_train.go` (7)

- Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Residual NormalBP leaf × all 34 dtypes — spectrum + speed
- Residual NormalBP volume 1×1×1×1 Float32 — exact bits
- Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Residual NormalBP volume 2×2×2×1 Float32 — exact bits
- Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rmsnorm.go` (6)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- RMSNorm Float32 [batch,dim] γ=1 — exact bits vs Go
- RMSNorm Float32 [batch,seq,dim] random γ — exact bits vs Go
- RMSNorm Float32 custom eps — exact bits vs Go
- RMSNorm FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `rmsnorm_train.go` (7)

- RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)
- RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rnn.go` (4)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- RNN Float32 — exact bits vs Go
- RNN FormatNone ×34 — ≥99% + Go/C++/Metal timings

#### `rnn_train.go` (7)

- RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- RNN NormalBP leaf × all 34 dtypes — spectrum + speed
- RNN NormalBP volume 1×1×1×1 Float32 — exact bits
- RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RNN NormalBP volume 2×2×2×1 Float32 — exact bits
- RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `seqmix.go` (4)

- Locate welvet.metal tool
- seqmix Kind iota + String — Go
- seqmix mixer tags — Go MHA/Mamba/GDN
- seqmix Kind names + tags — Go vs Metal WSX1

#### `sequential.go` (5)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Sequential Dense Float32 [B,D] — exact bits vs Go
- Sequential Dense Float32 [B,T,D] — exact bits vs Go
- Sequential Dense FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `sequential_train.go` (7)

- Sequential NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Sequential NormalBP leaf × all 34 dtypes — spectrum + speed
- Sequential NormalBP volume 1×1×1×1 Float32 — exact bits
- Sequential NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Sequential NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Sequential NormalBP volume 2×2×2×1 Float32 — exact bits
- Sequential NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `softmax.go` (11)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Softmax Standard Float32 — exact bits vs Go
- Softmax Temperature Float32 — exact bits vs Go
- Softmax Masked Float32 — exact bits vs Go
- Softmax Sparse Float32 — exact bits vs Go
- Softmax Entmax Float32 — exact bits vs Go
- Softmax Grid Float32 — exact bits vs Go
- Softmax Hierarchical Float32 — exact bits vs Go
- Softmax Gumbel Float32 — smoke (non-deterministic)
- Softmax all Kind ×34 — exact_bit + Go/C++/Metal timings

#### `softmax_train.go` (9)

- Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)
- Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)
- Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed
- Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum
- Softmax NormalBP volume 1×1×1×1 Float32 — exact bits
- Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Softmax NormalBP volume 2×2×2×1 Float32 — exact bits
- Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stack_train.go` (7)

- Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Stack NormalBP leaf × all 34 dtypes — spectrum + speed
- Stack NormalBP volume 1×1×1×1 Float32 — exact bits
- Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Stack NormalBP volume 2×2×2×1 Float32 — exact bits
- Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stepbp_train.go` (84)

- Dense StepBP multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense StepBP 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepBP 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm StepBP 1-hop × all 34 dtypes — spectrum
- LayerNorm StepBP 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm StepBP 1-hop × all 34 dtypes — spectrum
- Softmax StepBP 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax StepBP 1-hop × all 34 dtypes — spectrum
- SwiGLU StepBP 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU StepBP 1-hop × all 34 dtypes — spectrum
- Sequential StepBP 1-hop Float32 — exact tensors; Metal loss band
- Sequential StepBP 1-hop × all 34 dtypes — spectrum
- Residual StepBP 1-hop Float32 — exact tensors; Metal loss band
- Residual StepBP 1-hop × all 34 dtypes — spectrum
- Mamba StepBP 1-hop Float32 — exact tensors; Metal loss band
- Mamba StepBP 1-hop × all 34 dtypes — spectrum
- Metacognition StepBP 1-hop Float32 — exact tensors; Metal loss band
- Metacognition StepBP 1-hop × all 34 dtypes — spectrum
- CNN2 StepBP rank 4 × all 34 dtypes
- CNN3 StepBP rank 5 × all 34 dtypes
- ConvT2 StepBP rank 4 × all 34 dtypes
- ConvT3 StepBP rank 5 × all 34 dtypes
- Dense ModeMeshBP volume 211 × all 34 dtypes
- RMSNorm ModeMeshBP volume 211 × all 34 dtypes
- LayerNorm ModeMeshBP volume 211 × all 34 dtypes
- SwiGLU ModeMeshBP volume 211 × all 34 dtypes
- Softmax ModeMeshBP volume 211 × all 34 dtypes
- Embedding ModeMeshBP volume 211 × all 34 dtypes
- MHA ModeMeshBP volume 211 × all 34 dtypes
- RNN ModeMeshBP volume 211 × all 34 dtypes
- LSTM ModeMeshBP volume 211 × all 34 dtypes
- Sequential ModeMeshBP volume 211 × all 34 dtypes
- Residual ModeMeshBP volume 211 × all 34 dtypes
- Mamba ModeMeshBP volume 211 × all 34 dtypes
- Metacognition ModeMeshBP volume 211 × all 34 dtypes
- NestedStack ModeMeshBP volume 211 × all 34 dtypes
- CNN1 ModeMeshBP volume 211 × all 34 dtypes
- CNN2 ModeMeshBP volume 211 × all 34 dtypes
- CNN3 ModeMeshBP volume 211 × all 34 dtypes
- ConvT1 ModeMeshBP volume 211 × all 34 dtypes
- ConvT2 ModeMeshBP volume 211 × all 34 dtypes
- ConvT3 ModeMeshBP volume 211 × all 34 dtypes
- Dense MeshBP mesh-clock 111 × all 34 dtypes
- Dense MeshBP mesh-clock 222 × all 34 dtypes
- Dense MeshBP disabled cell and remote link
- GDN StepBP 1-hop Float32 — ≥strict vs Go (scan FP)
- NestedStack StepBP 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepBP 1-hop × all 34 dtypes — spectrum
- Embedding StepBP 1-hop Float32 — exact tensors; Metal loss band
- Embedding StepBP 1-hop × all 34 dtypes — spectrum
- MHA StepBP 1-hop Float32 — exact tensors; Metal loss band
- MHA StepBP 1-hop × all 34 dtypes — spectrum
- CNN1 StepBP 1-hop Float32 — exact tensors; Metal loss band
- CNN1 StepBP 1-hop × all 34 dtypes — spectrum
- ConvT1 StepBP 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 StepBP 1-hop × all 34 dtypes — spectrum
- RNN StepBP 1-hop Float32 — exact tensors; Metal loss band
- RNN StepBP 1-hop × all 34 dtypes — spectrum
- LSTM StepBP 1-hop Float32 — exact tensors; Metal loss band
- LSTM StepBP 1-hop × all 34 dtypes — spectrum
- Dense MeshBP 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshBP 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshBP 2×2×2×1 Float32 — exact tensors; Metal loss band
- Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshBP volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `steptween_train.go` (60)

- Dense StepTween multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense StepTween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTween 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm StepTween 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTween 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm StepTween 1-hop × all 34 dtypes — spectrum
- Softmax StepTween 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax StepTween 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTween 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU StepTween 1-hop × all 34 dtypes — spectrum
- Sequential StepTween 1-hop Float32 — exact tensors; Metal loss band
- Sequential StepTween 1-hop × all 34 dtypes — spectrum
- Residual StepTween 1-hop Float32 — exact tensors; Metal loss band
- Residual StepTween 1-hop × all 34 dtypes — spectrum
- Mamba StepTween 1-hop Float32 — exact tensors; Metal loss band
- Mamba StepTween 1-hop × all 34 dtypes — spectrum
- Embedding StepTween 1-hop Float32 — exact tensors; Metal loss band
- Embedding StepTween 1-hop × all 34 dtypes — spectrum
- MHA StepTween 1-hop Float32 — exact tensors; Metal loss band
- MHA StepTween 1-hop × all 34 dtypes — spectrum
- CNN1 StepTween 1-hop Float32 — exact tensors; Metal loss band
- CNN1 StepTween 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTween 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 StepTween 1-hop × all 34 dtypes — spectrum
- RNN StepTween 1-hop Float32 — exact tensors; Metal loss band
- RNN StepTween 1-hop × all 34 dtypes — spectrum
- LSTM StepTween 1-hop Float32 — exact tensors; Metal loss band
- LSTM StepTween 1-hop × all 34 dtypes — spectrum
- Metacognition StepTween Float32
- Metacognition StepTween × all 34 dtypes
- GDN StepTween Float32 — ≥strict weights; Metal loss band
- NestedStack StepTween Float32
- NestedStack StepTween × all 34 dtypes
- Metacognition StepTween volume 222 Float32
- GDN StepTween volume 222 Float32
- NestedStack StepTween volume 222 Float32
- CNN2 StepTween rank 4 × all 34 dtypes
- CNN3 StepTween rank 5 × all 34 dtypes
- ConvT2 StepTween rank 4 × all 34 dtypes
- ConvT3 StepTween rank 5 × all 34 dtypes
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `steptweenalt_train.go` (60)

- Dense StepTweenAlt multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense StepTweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax StepTweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Sequential StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Residual StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Residual StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Mamba StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Metacognition StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenAlt rank 4 × all 34 dtypes
- CNN3 StepTweenAlt rank 5 × all 34 dtypes
- ConvT2 StepTweenAlt rank 4 × all 34 dtypes
- ConvT3 StepTweenAlt rank 5 × all 34 dtypes
- GDN StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Embedding StepTweenAlt 1-hop × all 34 dtypes — spectrum
- MHA StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- MHA StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- CNN1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- RNN StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- RNN StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenAlt 1-hop Float32 — exact tensors; Metal loss band
- LSTM StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `steptweenchain_train.go` (60)

- Dense StepTweenChain multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenChain 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum
- Residual StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- Residual StepTweenChain 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenChain rank 4 × all 34 dtypes
- CNN3 StepTweenChain rank 5 × all 34 dtypes
- ConvT2 StepTweenChain rank 4 × all 34 dtypes
- ConvT3 StepTweenChain rank 5 × all 34 dtypes
- GDN StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum
- MHA StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- MHA StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- RNN StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- RNN StepTweenChain 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenChain 1-hop Float32 — exact tensors; Metal loss band
- LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `steptweensplit_train.go` (60)

- Dense StepTweenSplit multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense StepTweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax StepTweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Sequential StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Residual StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Mamba StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Metacognition StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenSplit rank 4 × all 34 dtypes
- CNN3 StepTweenSplit rank 5 × all 34 dtypes
- ConvT2 StepTweenSplit rank 4 × all 34 dtypes
- ConvT3 StepTweenSplit rank 5 × all 34 dtypes
- GDN StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- NestedStack StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Embedding StepTweenSplit 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- MHA StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- CNN1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- RNN StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplit 1-hop Float32 — exact tensors; Metal loss band
- LSTM StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `suite.go` (7)

- Locate welvet.metal welvet_dense_fwd
- Locate welvet.cpp welvet_dense_fwd (optional speed)
- Metal device identity
- Spectrum ladder self-check
- Dense forward identity+bias Float32 Metal — exact bits vs Go
- Dense forward ReLU Float32 Metal — exact bits vs Go
- Metal FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `swiglu.go` (6)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- SwiGLU Float32 [batch,in] — exact bits vs Go
- SwiGLU Float32 [batch,seq,in] — exact bits vs Go
- SwiGLU DefaultFFN Float32 — exact bits vs Go
- SwiGLU FormatNone ×34 — exact_bit + Go/C++/Metal timings

#### `swiglu_train.go` (7)

- SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed
- SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `tween_train.go` (80)

- Dense Tween multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense Tween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Tween 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm Tween 1-hop × all 34 dtypes — spectrum
- LayerNorm Tween 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm Tween 1-hop × all 34 dtypes — spectrum
- Softmax Tween 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax Tween 1-hop × all 34 dtypes — spectrum
- SwiGLU Tween 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU Tween 1-hop × all 34 dtypes — spectrum
- Sequential Tween 1-hop Float32 — exact tensors; Metal loss band
- Sequential Tween 1-hop × all 34 dtypes — spectrum
- Residual Tween 1-hop Float32 — exact tensors; Metal loss band
- Residual Tween 1-hop × all 34 dtypes — spectrum
- Mamba Tween 1-hop Float32 — exact tensors; Metal loss band
- Mamba Tween 1-hop × all 34 dtypes — spectrum
- Metacognition Tween 1-hop Float32 — exact tensors; Metal loss band
- Metacognition Tween 1-hop × all 34 dtypes — spectrum
- CNN2 Tween rank 4 × all 34 dtypes
- CNN3 Tween rank 5 × all 34 dtypes
- ConvT2 Tween rank 4 × all 34 dtypes
- ConvT3 Tween rank 5 × all 34 dtypes
- GDN Tween 1-hop Float32 — ≥strict weights; Metal loss band
- NestedStack Tween 1-hop Float32 — exact tensors; Metal loss band
- NestedStack Tween 1-hop × all 34 dtypes — spectrum
- Embedding Tween 1-hop Float32 — exact tensors; Metal loss band
- Embedding Tween 1-hop × all 34 dtypes — spectrum
- MHA Tween 1-hop Float32 — exact tensors; Metal loss band
- MHA Tween 1-hop × all 34 dtypes — spectrum
- CNN1 Tween 1-hop Float32 — exact tensors; Metal loss band
- CNN1 Tween 1-hop × all 34 dtypes — spectrum
- ConvT1 Tween 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 Tween 1-hop × all 34 dtypes — spectrum
- RNN Tween 1-hop Float32 — exact tensors; Metal loss band
- RNN Tween 1-hop × all 34 dtypes — spectrum
- LSTM Tween 1-hop Float32 — exact tensors; Metal loss band
- LSTM Tween 1-hop × all 34 dtypes — spectrum
- RMSNorm Tween volume 211 × all 34 dtypes
- LayerNorm Tween volume 211 × all 34 dtypes
- SwiGLU Tween volume 211 × all 34 dtypes
- Softmax Tween volume 211 × all 34 dtypes
- Embedding Tween volume 211 × all 34 dtypes
- MHA Tween volume 211 × all 34 dtypes
- RNN Tween volume 211 × all 34 dtypes
- LSTM Tween volume 211 × all 34 dtypes
- Seq Tween volume 211 × all 34 dtypes
- Re Tween volume 211 × all 34 dtypes
- Mb Tween volume 211 × all 34 dtypes
- Mc Tween volume 211 × all 34 dtypes
- Sk Tween volume 211 × all 34 dtypes
- CNN1 Tween volume 211 × all 34 dtypes
- CNN2 Tween volume 211 × all 34 dtypes
- CNN3 Tween volume 211 × all 34 dtypes
- ConvT1 Tween volume 211 × all 34 dtypes
- ConvT2 Tween volume 211 × all 34 dtypes
- ConvT3 Tween volume 211 × all 34 dtypes
- Dense Tween volume 211 × all 34 dtypes
- Dense Tween volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense Tween volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `tweenalt_train.go` (80)

- Dense TweenAlt multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense TweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax TweenAlt 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax TweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU TweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Sequential TweenAlt 1-hop × all 34 dtypes — spectrum
- Residual TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Residual TweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Mamba TweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Metacognition TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 TweenAlt rank 4 × all 34 dtypes
- CNN3 TweenAlt rank 5 × all 34 dtypes
- ConvT2 TweenAlt rank 4 × all 34 dtypes
- ConvT3 TweenAlt rank 5 × all 34 dtypes
- GDN TweenAlt 1-hop Float32 — ≥strict weights; finite GDN loss band
- NestedStack TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- NestedStack TweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- Embedding TweenAlt 1-hop × all 34 dtypes — spectrum
- MHA TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- MHA TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- CNN1 TweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 TweenAlt 1-hop × all 34 dtypes — spectrum
- RNN TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- RNN TweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM TweenAlt 1-hop Float32 — exact tensors; Metal loss band
- LSTM TweenAlt 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenAlt volume 211 × all 34 dtypes
- LayerNorm TweenAlt volume 211 × all 34 dtypes
- SwiGLU TweenAlt volume 211 × all 34 dtypes
- Softmax TweenAlt volume 211 × all 34 dtypes
- Embedding TweenAlt volume 211 × all 34 dtypes
- MHA TweenAlt volume 211 × all 34 dtypes
- RNN TweenAlt volume 211 × all 34 dtypes
- LSTM TweenAlt volume 211 × all 34 dtypes
- Seq TweenAlt volume 211 × all 34 dtypes
- Re TweenAlt volume 211 × all 34 dtypes
- Mb TweenAlt volume 211 × all 34 dtypes
- Mc TweenAlt volume 211 × all 34 dtypes
- Sk TweenAlt volume 211 × all 34 dtypes
- CNN1 TweenAlt volume 211 × all 34 dtypes
- CNN2 TweenAlt volume 211 × all 34 dtypes
- CNN3 TweenAlt volume 211 × all 34 dtypes
- ConvT1 TweenAlt volume 211 × all 34 dtypes
- ConvT2 TweenAlt volume 211 × all 34 dtypes
- ConvT3 TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense TweenAlt volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM TweenAlt volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `tweenchain_train.go` (80)

- Dense TweenChain multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense TweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenChain 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm TweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenChain 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm TweenChain 1-hop × all 34 dtypes — spectrum
- Softmax TweenChain 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax TweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenChain 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU TweenChain 1-hop × all 34 dtypes — spectrum
- Sequential TweenChain 1-hop Float32 — exact tensors; Metal loss band
- Sequential TweenChain 1-hop × all 34 dtypes — spectrum
- Residual TweenChain 1-hop Float32 — exact tensors; Metal loss band
- Residual TweenChain 1-hop × all 34 dtypes — spectrum
- Mamba TweenChain 1-hop Float32 — exact tensors; Metal loss band
- Mamba TweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition TweenChain 1-hop Float32 — exact tensors; Metal loss band
- Metacognition TweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 TweenChain rank 4 × all 34 dtypes
- CNN3 TweenChain rank 5 × all 34 dtypes
- ConvT2 TweenChain rank 4 × all 34 dtypes
- ConvT3 TweenChain rank 5 × all 34 dtypes
- GDN TweenChain 1-hop Float32 — ≥strict weights; Metal loss band
- NestedStack TweenChain 1-hop Float32 — exact tensors; Metal loss band
- NestedStack TweenChain 1-hop × all 34 dtypes — spectrum
- Embedding TweenChain 1-hop Float32 — exact tensors; Metal loss band
- Embedding TweenChain 1-hop × all 34 dtypes — spectrum
- MHA TweenChain 1-hop Float32 — exact tensors; Metal loss band
- MHA TweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 TweenChain 1-hop Float32 — exact tensors; Metal loss band
- CNN1 TweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenChain 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 TweenChain 1-hop × all 34 dtypes — spectrum
- RNN TweenChain 1-hop Float32 — exact tensors; Metal loss band
- RNN TweenChain 1-hop × all 34 dtypes — spectrum
- LSTM TweenChain 1-hop Float32 — exact tensors; Metal loss band
- LSTM TweenChain 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenChain volume 211 × all 34 dtypes
- LayerNorm TweenChain volume 211 × all 34 dtypes
- SwiGLU TweenChain volume 211 × all 34 dtypes
- Softmax TweenChain volume 211 × all 34 dtypes
- Embedding TweenChain volume 211 × all 34 dtypes
- MHA TweenChain volume 211 × all 34 dtypes
- RNN TweenChain volume 211 × all 34 dtypes
- LSTM TweenChain volume 211 × all 34 dtypes
- Seq TweenChain volume 211 × all 34 dtypes
- Re TweenChain volume 211 × all 34 dtypes
- Mb TweenChain volume 211 × all 34 dtypes
- Mc TweenChain volume 211 × all 34 dtypes
- Sk TweenChain volume 211 × all 34 dtypes
- CNN1 TweenChain volume 211 × all 34 dtypes
- CNN2 TweenChain volume 211 × all 34 dtypes
- CNN3 TweenChain volume 211 × all 34 dtypes
- ConvT1 TweenChain volume 211 × all 34 dtypes
- ConvT2 TweenChain volume 211 × all 34 dtypes
- ConvT3 TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense TweenChain volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM TweenChain volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `tweensplit_train.go` (80)

- Dense TweenSplit multi-hop Float32 — exact tensors; Metal loss band (W/loss)
- Dense TweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- RMSNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- LayerNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplit 1-hop Float32 — exact tensors; Metal loss band (loss)
- Softmax TweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- SwiGLU TweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Sequential TweenSplit 1-hop × all 34 dtypes — spectrum
- Residual TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Residual TweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Mamba TweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Metacognition TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 TweenSplit rank 4 × all 34 dtypes
- CNN3 TweenSplit rank 5 × all 34 dtypes
- ConvT2 TweenSplit rank 4 × all 34 dtypes
- ConvT3 TweenSplit rank 5 × all 34 dtypes
- GDN TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- NestedStack TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- NestedStack TweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- Embedding TweenSplit 1-hop × all 34 dtypes — spectrum
- MHA TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- MHA TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- CNN1 TweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- ConvT1 TweenSplit 1-hop × all 34 dtypes — spectrum
- RNN TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- RNN TweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplit 1-hop Float32 — exact tensors; Metal loss band
- LSTM TweenSplit 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenSplit volume 211 × all 34 dtypes
- LayerNorm TweenSplit volume 211 × all 34 dtypes
- SwiGLU TweenSplit volume 211 × all 34 dtypes
- Softmax TweenSplit volume 211 × all 34 dtypes
- Embedding TweenSplit volume 211 × all 34 dtypes
- MHA TweenSplit volume 211 × all 34 dtypes
- RNN TweenSplit volume 211 × all 34 dtypes
- LSTM TweenSplit volume 211 × all 34 dtypes
- Seq TweenSplit volume 211 × all 34 dtypes
- Re TweenSplit volume 211 × all 34 dtypes
- Mb TweenSplit volume 211 × all 34 dtypes
- Mc TweenSplit volume 211 × all 34 dtypes
- Sk TweenSplit volume 211 × all 34 dtypes
- CNN1 TweenSplit volume 211 × all 34 dtypes
- CNN2 TweenSplit volume 211 × all 34 dtypes
- CNN3 TweenSplit volume 211 × all 34 dtypes
- ConvT1 TweenSplit volume 211 × all 34 dtypes
- ConvT2 TweenSplit volume 211 × all 34 dtypes
- ConvT3 TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 1×1×1×1 Float32 — exact tensors; Metal loss band
- Dense TweenSplit volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band
- Dense TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RMSNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LayerNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Softmax TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- SwiGLU TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Sequential TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Residual TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Mamba TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Metacognition TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- GDN TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- NestedStack TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- Embedding TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- MHA TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- CNN3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- ConvT3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- RNN TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band
- LSTM TweenSplit volume 2×2×2×1 Float32 — exact tensors; Metal loss band

#### `volume.go` (22)

- Locate welvet.metal tool
- Locate welvet.cpp tool (optional speed)
- Volume Dense 1×1×1×2 chain — exact bits vs Go
- Volume Dense 2×1×1×1 remote hop — exact bits vs Go
- Volume Dense 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume MHA 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume SwiGLU 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume Sequential 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume Residual 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume RNN 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume LSTM 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume Metacognition 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume Embedding 1×1×1×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume Mamba 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume GDN 3×3×3×1 Float32 — ≥99% + Go/C++/Metal timings
- Volume CNN1 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume CNN2 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume CNN3 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume ConvT1 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume ConvT2 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume ConvT3 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings
- Volume AllLayers 3×3×3 × dtypes — mount-all volume wires

### suites/go_v_mlx (22 leaf cases)

#### `suite.go` (22)

- Caps — README MLX layers × float32
- Dense caps — all compute dtypes (+ float64 rejected)
- Dense f32 B=32 in=512 out=512 — transfer/prep/exec/e2e + accuracy
- Dense ALL dtypes — Session prepare/exec vs Go
- Dense Go layer — SyncToMLX + BackendMLX (f32/f16/bf16)
- Softmax f32 64×4096 — transfer/prep/exec/e2e + accuracy
- Softmax ALL float dtypes — Session × f32/f16/bf16
- RMSNorm f32 [8,128,512] — transfer/prep/exec/e2e + accuracy
- RMSNorm ALL float dtypes — Session × f32/f16/bf16
- LayerNorm f32 [8,128,512] — transfer/prep/exec/e2e + accuracy
- LayerNorm ALL float dtypes — Session × f32/f16/bf16
- CNN2 ALL dtypes — Session × floats+ints
- CNN3 ALL dtypes — Session × floats+ints
- ConvT2 ALL dtypes — Session × floats+ints
- ConvT3 ALL dtypes — Session × floats+ints
- LSTM ALL float dtypes — Session × f32/f16/bf16
- Embedding ALL dtypes — Session × floats+ints
- Residual f32 n=65536 — transfer/prep/exec/e2e + accuracy
- Residual ALL dtypes — Session × floats+ints+c64
- SwiGLU ALL float dtypes — Session × f32/f16/bf16
- MHA ALL float dtypes — Session × f32/f16/bf16
- Sequential Dense tower dim=64 depth=3 — go/mlx f32

### suites/go_v_qnn (16 leaf cases)

#### `suite.go` (16)

- Matrix — all README layers × dtypes (writes RESULTS.md)
- Caps — README HTP layers × float32
- Dense f32 B=64 in=1024 out=1024 — transfer/prep/exec/e2e + accuracy
- Softmax f32 128×8192 — transfer/prep/exec/e2e + accuracy
- RMSNorm f32 [16,256,1024] — transfer/prep/exec/e2e + accuracy
- LayerNorm f32 [16,256,1024] — transfer/prep/exec/e2e + accuracy
- CNN2 f32 8×128×128×64→64 — transfer/prep/exec/e2e + accuracy
- CNN3 f32 2×16³×32→32 — transfer/prep/exec/e2e
- ConvT2 f32 8×64×64×64→64 — transfer/prep/exec/e2e
- ConvT3 f32 2×16³×32→32 — transfer/prep/exec/e2e
- LSTM f32 T=8 B=8 in=128 hid=128 — transfer/prep/exec/e2e
- Embedding f32 vocab=16k dim=1024 n=1024 — transfer/prep/exec/e2e + accuracy
- Residual f32 n=256k — transfer/prep/exec/e2e + accuracy
- SwiGLU f32 B=16 in=512 hidden=1408 — transfer/prep/exec/e2e
- MHA f32 B=2 S=64 D=256 H=4 — transfer/prep/exec/e2e
- Sequential Dense tower dim=128 depth=3 — go/native/qnn exec-only

### suites/go_v_rocm (939 leaf cases)

#### `cnn.go` (8)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- CNN1 Float32 Linear pad — exact bits vs Go
- CNN2 Float32 ReLU — exact bits vs Go
- CNN3 Float32 Linear — exact bits vs Go
- CNN1 FormatNone ×34 — exact_bit + Go/C++/ROCm timings
- CNN2 FormatNone ×34 — exact_bit + Go/C++/ROCm timings
- CNN3 FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `convt.go` (8)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- ConvT1 Float32 Linear pad — exact bits vs Go
- ConvT2 Float32 ReLU — exact bits vs Go
- ConvT3 Float32 Linear out_pad — exact bits vs Go
- ConvT1 FormatNone ×34 — exact_bit + Go/C++/ROCm timings
- ConvT2 FormatNone ×34 — exact_bit + Go/C++/ROCm timings
- ConvT3 FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `dense_train.go` (7)

- Dense NormalBP leaf Float32 ReLU — exact tensors; ROCm loss band (dX/dW/W/loss)
- Dense NormalBP leaf × all 34 dtypes — spectrum + speed
- Dense NormalBP volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense NormalBP volume 1×1×1×2 chain Float32 — exact tensors; ROCm loss band
- Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Dense NormalBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `embedding.go` (6)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Embedding Float32 Linear [B,T] — exact bits vs Go
- Embedding Float32 Linear [T] rank1 — exact bits vs Go
- Embedding Float32 Linear [B,T,1] rank3 — exact bits vs Go
- Embedding FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `embedding_train.go` (7)

- Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Embedding NormalBP leaf × all 34 dtypes — spectrum + speed
- Embedding NormalBP volume 1×1×1×1 Float32 — exact bits
- Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Embedding NormalBP volume 2×2×2×1 Float32 — exact bits
- Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `gdn.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- GDN Float32 primary — ≥industry vs Go, exact vs C++
- GDN Float32 alt GQA — ≥industry vs Go, exact vs C++
- GDN Float32 mid — parity + Go/C++/ROCm timings

#### `gdn_train.go` (4)

- GDN NormalBP leaf Float32 — ≥loose vs Go (+ speed)
- GDN NormalBP volume 1×1×1×1 Float32 — ≥industry vs Go
- GDN NormalBP volume 1×1×1×2 chain Float32 — ≥industry vs Go
- GDN NormalBP volume 2×2×2×1 Float32 — ≥industry vs Go

#### `kmeans.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- KMeans probabilities Float32 Linear — exact bits vs Go
- KMeans features Float32 Linear — exact bits vs Go
- KMeans both modes ×34 — exact_bit + Go/C++/ROCm timings

#### `kmeans_train.go` (8)

- KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed
- KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits
- KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `layernorm.go` (6)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits vs Go
- LayerNorm Float32 [batch,seq,dim] random affine — exact bits vs Go
- LayerNorm Float32 custom eps — exact bits vs Go
- LayerNorm FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `layernorm_train.go` (7)

- LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)
- LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `lstm.go` (4)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- LSTM Float32 — exact bits vs Go
- LSTM FormatNone ×34 — ≥98% + Go/C++/ROCm timings

#### `lstm_train.go` (7)

- LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- LSTM NormalBP leaf × all 34 dtypes — spectrum + speed
- LSTM NormalBP volume 1×1×1×1 Float32 — exact bits
- LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LSTM NormalBP volume 2×2×2×1 Float32 — exact bits
- LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mamba.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Mamba Float32 [B,T,D] — exact bits vs Go
- Mamba Float32 alt shape — exact bits vs Go
- Mamba FormatNone ×34 — ≥99% + Go/C++/ROCm timings

#### `mamba_train.go` (7)

- Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Mamba NormalBP leaf × all 34 dtypes — spectrum + speed
- Mamba NormalBP volume 1×1×1×1 Float32 — exact bits
- Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Mamba NormalBP volume 2×2×2×1 Float32 — exact bits
- Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `meshtween_train.go` (28)

- Dense MeshTween mesh-clock 111 × all 34 dtypes
- Dense MeshTween mesh-clock 211 × all 34 dtypes
- Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `meshtweenchain_train.go` (28)

- Dense MeshTweenChain mesh-clock 111 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 211 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `metacognition.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Metacognition Float32 [B,D] — exact bits vs Go
- Metacognition Float32 [B,T,D] — exact bits vs Go
- Metacognition ×34 — ≥99% + Go/C++/ROCm timings

#### `metacognition_train.go` (7)

- Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed
- Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits
- Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mha.go` (6)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- MHA DecoderCausal Float32 — exact bits vs Go
- MHA GQA 4h/2kv Float32 — exact bits vs Go
- MHA Encoder bidirectional Float32 — exact bits vs Go
- MHA DecoderCausal FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `mha_train.go` (8)

- MHA NormalBP leaf DecoderCausal Float32 — exact bits
- MHA NormalBP leaf GQA + Encoder Float32 — exact bits
- MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed
- MHA NormalBP volume 1×1×1×1 Float32 — exact bits
- MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits
- MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- MHA NormalBP volume 2×2×2×1 Float32 — exact bits
- MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `parallel.go` (11)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Parallel Add Float32 [B,D] — exact bits vs Go
- Parallel Add Float32 [B,T,D] — exact bits vs Go
- Parallel Concat Float32 — exact bits vs Go
- Parallel Avg Float32 — exact bits vs Go
- Parallel Filter Float32 — exact bits vs Go
- Parallel Max Float32 — exact bits vs Go
- Parallel SparseK Float32 — exact bits vs Go
- Parallel Disagree Float32 (2+3 branches) — exact bits vs Go
- Parallel Add ×34 — ≥99% + Go/C++/ROCm timings

#### `parallel_nbp_train.go` (9)

- Dense Add Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP × all 34 dtypes — spectrum
- Dense Avg Parallel NormalBP Float32 — exact bits
- Dense Concat Parallel NormalBP Float32 — exact bits
- Dense Max Parallel NormalBP Float32 — exact bits
- Dense SparseK Parallel NormalBP Float32 — exact bits
- Dense Disagree Parallel NormalBP Float32 — exact bits
- Dense Filter Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits

#### `residual.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Residual Dense Float32 [B,D] — exact bits vs Go
- Residual Dense Float32 [B,T,D] — exact bits vs Go
- Residual Dense FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `residual_train.go` (7)

- Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Residual NormalBP leaf × all 34 dtypes — spectrum + speed
- Residual NormalBP volume 1×1×1×1 Float32 — exact bits
- Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Residual NormalBP volume 2×2×2×1 Float32 — exact bits
- Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rmsnorm.go` (6)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- RMSNorm Float32 [batch,dim] γ=1 — exact bits vs Go
- RMSNorm Float32 [batch,seq,dim] random γ — exact bits vs Go
- RMSNorm Float32 custom eps — exact bits vs Go
- RMSNorm FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `rmsnorm_train.go` (7)

- RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)
- RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rnn.go` (4)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- RNN Float32 — exact bits vs Go
- RNN FormatNone ×34 — ≥99% + Go/C++/ROCm timings

#### `rnn_train.go` (7)

- RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- RNN NormalBP leaf × all 34 dtypes — spectrum + speed
- RNN NormalBP volume 1×1×1×1 Float32 — exact bits
- RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RNN NormalBP volume 2×2×2×1 Float32 — exact bits
- RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `seqmix.go` (4)

- Locate welvet.rocm tool
- seqmix Kind iota + String — Go
- seqmix mixer tags — Go MHA/Mamba/GDN
- seqmix Kind names + tags — Go vs ROCm WSX1

#### `sequential.go` (5)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Sequential Dense Float32 [B,D] — exact bits vs Go
- Sequential Dense Float32 [B,T,D] — exact bits vs Go
- Sequential Dense FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `sequential_train.go` (7)

- Sequential NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Sequential NormalBP leaf × all 34 dtypes — spectrum + speed
- Sequential NormalBP volume 1×1×1×1 Float32 — exact bits
- Sequential NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Sequential NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Sequential NormalBP volume 2×2×2×1 Float32 — exact bits
- Sequential NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `softmax.go` (11)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- Softmax Standard Float32 — exact bits vs Go
- Softmax Temperature Float32 — exact bits vs Go
- Softmax Masked Float32 — exact bits vs Go
- Softmax Sparse Float32 — exact bits vs Go
- Softmax Entmax Float32 — exact bits vs Go
- Softmax Grid Float32 — exact bits vs Go
- Softmax Hierarchical Float32 — exact bits vs Go
- Softmax Gumbel Float32 — smoke (non-deterministic)
- Softmax all Kind ×34 — exact_bit + Go/C++/ROCm timings

#### `softmax_train.go` (9)

- Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)
- Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)
- Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed
- Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum
- Softmax NormalBP volume 1×1×1×1 Float32 — exact bits
- Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Softmax NormalBP volume 2×2×2×1 Float32 — exact bits
- Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stack_train.go` (7)

- Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Stack NormalBP leaf × all 34 dtypes — spectrum + speed
- Stack NormalBP volume 1×1×1×1 Float32 — exact bits
- Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Stack NormalBP volume 2×2×2×1 Float32 — exact bits
- Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stepbp_train.go` (84)

- Dense StepBP multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense StepBP 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepBP 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm StepBP 1-hop × all 34 dtypes — spectrum
- LayerNorm StepBP 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm StepBP 1-hop × all 34 dtypes — spectrum
- Softmax StepBP 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax StepBP 1-hop × all 34 dtypes — spectrum
- SwiGLU StepBP 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU StepBP 1-hop × all 34 dtypes — spectrum
- Sequential StepBP 1-hop Float32 — exact tensors; ROCm loss band
- Sequential StepBP 1-hop × all 34 dtypes — spectrum
- Residual StepBP 1-hop Float32 — exact tensors; ROCm loss band
- Residual StepBP 1-hop × all 34 dtypes — spectrum
- Mamba StepBP 1-hop Float32 — exact tensors; ROCm loss band
- Mamba StepBP 1-hop × all 34 dtypes — spectrum
- Metacognition StepBP 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition StepBP 1-hop × all 34 dtypes — spectrum
- CNN2 StepBP rank 4 × all 34 dtypes
- CNN3 StepBP rank 5 × all 34 dtypes
- ConvT2 StepBP rank 4 × all 34 dtypes
- ConvT3 StepBP rank 5 × all 34 dtypes
- Dense ModeMeshBP volume 211 × all 34 dtypes
- RMSNorm ModeMeshBP volume 211 × all 34 dtypes
- LayerNorm ModeMeshBP volume 211 × all 34 dtypes
- SwiGLU ModeMeshBP volume 211 × all 34 dtypes
- Softmax ModeMeshBP volume 211 × all 34 dtypes
- Embedding ModeMeshBP volume 211 × all 34 dtypes
- MHA ModeMeshBP volume 211 × all 34 dtypes
- RNN ModeMeshBP volume 211 × all 34 dtypes
- LSTM ModeMeshBP volume 211 × all 34 dtypes
- Sequential ModeMeshBP volume 211 × all 34 dtypes
- Residual ModeMeshBP volume 211 × all 34 dtypes
- Mamba ModeMeshBP volume 211 × all 34 dtypes
- Metacognition ModeMeshBP volume 211 × all 34 dtypes
- NestedStack ModeMeshBP volume 211 × all 34 dtypes
- CNN1 ModeMeshBP volume 211 × all 34 dtypes
- CNN2 ModeMeshBP volume 211 × all 34 dtypes
- CNN3 ModeMeshBP volume 211 × all 34 dtypes
- ConvT1 ModeMeshBP volume 211 × all 34 dtypes
- ConvT2 ModeMeshBP volume 211 × all 34 dtypes
- ConvT3 ModeMeshBP volume 211 × all 34 dtypes
- Dense MeshBP mesh-clock 111 × all 34 dtypes
- Dense MeshBP mesh-clock 222 × all 34 dtypes
- Dense MeshBP disabled cell and remote link
- GDN StepBP 1-hop Float32 — ≥strict vs Go (scan FP)
- NestedStack StepBP 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepBP 1-hop × all 34 dtypes — spectrum
- Embedding StepBP 1-hop Float32 — exact tensors; ROCm loss band
- Embedding StepBP 1-hop × all 34 dtypes — spectrum
- MHA StepBP 1-hop Float32 — exact tensors; ROCm loss band
- MHA StepBP 1-hop × all 34 dtypes — spectrum
- CNN1 StepBP 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 StepBP 1-hop × all 34 dtypes — spectrum
- ConvT1 StepBP 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 StepBP 1-hop × all 34 dtypes — spectrum
- RNN StepBP 1-hop Float32 — exact tensors; ROCm loss band
- RNN StepBP 1-hop × all 34 dtypes — spectrum
- LSTM StepBP 1-hop Float32 — exact tensors; ROCm loss band
- LSTM StepBP 1-hop × all 34 dtypes — spectrum
- Dense MeshBP 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshBP 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshBP 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshBP volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `steptween_train.go` (60)

- Dense StepTween multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense StepTween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTween 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm StepTween 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTween 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm StepTween 1-hop × all 34 dtypes — spectrum
- Softmax StepTween 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax StepTween 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTween 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU StepTween 1-hop × all 34 dtypes — spectrum
- Sequential StepTween 1-hop Float32 — exact tensors; ROCm loss band
- Sequential StepTween 1-hop × all 34 dtypes — spectrum
- Residual StepTween 1-hop Float32 — exact tensors; ROCm loss band
- Residual StepTween 1-hop × all 34 dtypes — spectrum
- Mamba StepTween 1-hop Float32 — exact tensors; ROCm loss band
- Mamba StepTween 1-hop × all 34 dtypes — spectrum
- Embedding StepTween 1-hop Float32 — exact tensors; ROCm loss band
- Embedding StepTween 1-hop × all 34 dtypes — spectrum
- MHA StepTween 1-hop Float32 — exact tensors; ROCm loss band
- MHA StepTween 1-hop × all 34 dtypes — spectrum
- CNN1 StepTween 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 StepTween 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTween 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 StepTween 1-hop × all 34 dtypes — spectrum
- RNN StepTween 1-hop Float32 — exact tensors; ROCm loss band
- RNN StepTween 1-hop × all 34 dtypes — spectrum
- LSTM StepTween 1-hop Float32 — exact tensors; ROCm loss band
- LSTM StepTween 1-hop × all 34 dtypes — spectrum
- Metacognition StepTween Float32
- Metacognition StepTween × all 34 dtypes
- GDN StepTween Float32 — ≥strict weights; ROCm loss band
- NestedStack StepTween Float32
- NestedStack StepTween × all 34 dtypes
- Metacognition StepTween volume 222 Float32
- GDN StepTween volume 222 Float32
- NestedStack StepTween volume 222 Float32
- CNN2 StepTween rank 4 × all 34 dtypes
- CNN3 StepTween rank 5 × all 34 dtypes
- ConvT2 StepTween rank 4 × all 34 dtypes
- ConvT3 StepTween rank 5 × all 34 dtypes
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `steptweenalt_train.go` (60)

- Dense StepTweenAlt multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense StepTweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax StepTweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Sequential StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Residual StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Residual StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Mamba StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenAlt rank 4 × all 34 dtypes
- CNN3 StepTweenAlt rank 5 × all 34 dtypes
- ConvT2 StepTweenAlt rank 4 × all 34 dtypes
- ConvT3 StepTweenAlt rank 5 × all 34 dtypes
- GDN StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Embedding StepTweenAlt 1-hop × all 34 dtypes — spectrum
- MHA StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- MHA StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- RNN StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- RNN StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- LSTM StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `steptweenchain_train.go` (60)

- Dense StepTweenChain multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum
- Residual StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Residual StepTweenChain 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenChain rank 4 × all 34 dtypes
- CNN3 StepTweenChain rank 5 × all 34 dtypes
- ConvT2 StepTweenChain rank 4 × all 34 dtypes
- ConvT3 StepTweenChain rank 5 × all 34 dtypes
- GDN StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum
- MHA StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- MHA StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- RNN StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- RNN StepTweenChain 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenChain 1-hop Float32 — exact tensors; ROCm loss band
- LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `steptweensplit_train.go` (60)

- Dense StepTweenSplit multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense StepTweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax StepTweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Sequential StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Residual StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Mamba StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenSplit rank 4 × all 34 dtypes
- CNN3 StepTweenSplit rank 5 × all 34 dtypes
- ConvT2 StepTweenSplit rank 4 × all 34 dtypes
- ConvT3 StepTweenSplit rank 5 × all 34 dtypes
- GDN StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Embedding StepTweenSplit 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- MHA StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- RNN StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- LSTM StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `suite.go` (7)

- Locate welvet.rocm welvet_dense_fwd
- Locate welvet.cpp welvet_dense_fwd (optional speed)
- ROCm device identity
- Spectrum ladder self-check
- Dense forward identity+bias Float32 ROCm — exact bits vs Go
- Dense forward ReLU Float32 ROCm — exact bits vs Go
- ROCm FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `swiglu.go` (6)

- Locate welvet.rocm tool
- Locate welvet.cpp tool (optional speed)
- SwiGLU Float32 [batch,in] — exact bits vs Go
- SwiGLU Float32 [batch,seq,in] — exact bits vs Go
- SwiGLU DefaultFFN Float32 — exact bits vs Go
- SwiGLU FormatNone ×34 — exact_bit + Go/C++/ROCm timings

#### `swiglu_train.go` (7)

- SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed
- SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `tween_train.go` (80)

- Dense Tween multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense Tween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Tween 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm Tween 1-hop × all 34 dtypes — spectrum
- LayerNorm Tween 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm Tween 1-hop × all 34 dtypes — spectrum
- Softmax Tween 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax Tween 1-hop × all 34 dtypes — spectrum
- SwiGLU Tween 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU Tween 1-hop × all 34 dtypes — spectrum
- Sequential Tween 1-hop Float32 — exact tensors; ROCm loss band
- Sequential Tween 1-hop × all 34 dtypes — spectrum
- Residual Tween 1-hop Float32 — exact tensors; ROCm loss band
- Residual Tween 1-hop × all 34 dtypes — spectrum
- Mamba Tween 1-hop Float32 — exact tensors; ROCm loss band
- Mamba Tween 1-hop × all 34 dtypes — spectrum
- Metacognition Tween 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition Tween 1-hop × all 34 dtypes — spectrum
- CNN2 Tween rank 4 × all 34 dtypes
- CNN3 Tween rank 5 × all 34 dtypes
- ConvT2 Tween rank 4 × all 34 dtypes
- ConvT3 Tween rank 5 × all 34 dtypes
- GDN Tween 1-hop Float32 — ≥strict weights; ROCm loss band
- NestedStack Tween 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack Tween 1-hop × all 34 dtypes — spectrum
- Embedding Tween 1-hop Float32 — exact tensors; ROCm loss band
- Embedding Tween 1-hop × all 34 dtypes — spectrum
- MHA Tween 1-hop Float32 — exact tensors; ROCm loss band
- MHA Tween 1-hop × all 34 dtypes — spectrum
- CNN1 Tween 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 Tween 1-hop × all 34 dtypes — spectrum
- ConvT1 Tween 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 Tween 1-hop × all 34 dtypes — spectrum
- RNN Tween 1-hop Float32 — exact tensors; ROCm loss band
- RNN Tween 1-hop × all 34 dtypes — spectrum
- LSTM Tween 1-hop Float32 — exact tensors; ROCm loss band
- LSTM Tween 1-hop × all 34 dtypes — spectrum
- RMSNorm Tween volume 211 × all 34 dtypes
- LayerNorm Tween volume 211 × all 34 dtypes
- SwiGLU Tween volume 211 × all 34 dtypes
- Softmax Tween volume 211 × all 34 dtypes
- Embedding Tween volume 211 × all 34 dtypes
- MHA Tween volume 211 × all 34 dtypes
- RNN Tween volume 211 × all 34 dtypes
- LSTM Tween volume 211 × all 34 dtypes
- Seq Tween volume 211 × all 34 dtypes
- Re Tween volume 211 × all 34 dtypes
- Mb Tween volume 211 × all 34 dtypes
- Mc Tween volume 211 × all 34 dtypes
- Sk Tween volume 211 × all 34 dtypes
- CNN1 Tween volume 211 × all 34 dtypes
- CNN2 Tween volume 211 × all 34 dtypes
- CNN3 Tween volume 211 × all 34 dtypes
- ConvT1 Tween volume 211 × all 34 dtypes
- ConvT2 Tween volume 211 × all 34 dtypes
- ConvT3 Tween volume 211 × all 34 dtypes
- Dense Tween volume 211 × all 34 dtypes
- Dense Tween volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense Tween volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM Tween volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `tweenalt_train.go` (80)

- Dense TweenAlt multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense TweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax TweenAlt 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax TweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU TweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Sequential TweenAlt 1-hop × all 34 dtypes — spectrum
- Residual TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Residual TweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Mamba TweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 TweenAlt rank 4 × all 34 dtypes
- CNN3 TweenAlt rank 5 × all 34 dtypes
- ConvT2 TweenAlt rank 4 × all 34 dtypes
- ConvT3 TweenAlt rank 5 × all 34 dtypes
- GDN TweenAlt 1-hop Float32 — ≥strict weights; finite GDN loss band
- NestedStack TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack TweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- Embedding TweenAlt 1-hop × all 34 dtypes — spectrum
- MHA TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- MHA TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 TweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 TweenAlt 1-hop × all 34 dtypes — spectrum
- RNN TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- RNN TweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM TweenAlt 1-hop Float32 — exact tensors; ROCm loss band
- LSTM TweenAlt 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenAlt volume 211 × all 34 dtypes
- LayerNorm TweenAlt volume 211 × all 34 dtypes
- SwiGLU TweenAlt volume 211 × all 34 dtypes
- Softmax TweenAlt volume 211 × all 34 dtypes
- Embedding TweenAlt volume 211 × all 34 dtypes
- MHA TweenAlt volume 211 × all 34 dtypes
- RNN TweenAlt volume 211 × all 34 dtypes
- LSTM TweenAlt volume 211 × all 34 dtypes
- Seq TweenAlt volume 211 × all 34 dtypes
- Re TweenAlt volume 211 × all 34 dtypes
- Mb TweenAlt volume 211 × all 34 dtypes
- Mc TweenAlt volume 211 × all 34 dtypes
- Sk TweenAlt volume 211 × all 34 dtypes
- CNN1 TweenAlt volume 211 × all 34 dtypes
- CNN2 TweenAlt volume 211 × all 34 dtypes
- CNN3 TweenAlt volume 211 × all 34 dtypes
- ConvT1 TweenAlt volume 211 × all 34 dtypes
- ConvT2 TweenAlt volume 211 × all 34 dtypes
- ConvT3 TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense TweenAlt volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM TweenAlt volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `tweenchain_train.go` (80)

- Dense TweenChain multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense TweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm TweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm TweenChain 1-hop × all 34 dtypes — spectrum
- Softmax TweenChain 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax TweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU TweenChain 1-hop × all 34 dtypes — spectrum
- Sequential TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Sequential TweenChain 1-hop × all 34 dtypes — spectrum
- Residual TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Residual TweenChain 1-hop × all 34 dtypes — spectrum
- Mamba TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Mamba TweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition TweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 TweenChain rank 4 × all 34 dtypes
- CNN3 TweenChain rank 5 × all 34 dtypes
- ConvT2 TweenChain rank 4 × all 34 dtypes
- ConvT3 TweenChain rank 5 × all 34 dtypes
- GDN TweenChain 1-hop Float32 — ≥strict weights; ROCm loss band
- NestedStack TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack TweenChain 1-hop × all 34 dtypes — spectrum
- Embedding TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- Embedding TweenChain 1-hop × all 34 dtypes — spectrum
- MHA TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- MHA TweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 TweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 TweenChain 1-hop × all 34 dtypes — spectrum
- RNN TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- RNN TweenChain 1-hop × all 34 dtypes — spectrum
- LSTM TweenChain 1-hop Float32 — exact tensors; ROCm loss band
- LSTM TweenChain 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenChain volume 211 × all 34 dtypes
- LayerNorm TweenChain volume 211 × all 34 dtypes
- SwiGLU TweenChain volume 211 × all 34 dtypes
- Softmax TweenChain volume 211 × all 34 dtypes
- Embedding TweenChain volume 211 × all 34 dtypes
- MHA TweenChain volume 211 × all 34 dtypes
- RNN TweenChain volume 211 × all 34 dtypes
- LSTM TweenChain volume 211 × all 34 dtypes
- Seq TweenChain volume 211 × all 34 dtypes
- Re TweenChain volume 211 × all 34 dtypes
- Mb TweenChain volume 211 × all 34 dtypes
- Mc TweenChain volume 211 × all 34 dtypes
- Sk TweenChain volume 211 × all 34 dtypes
- CNN1 TweenChain volume 211 × all 34 dtypes
- CNN2 TweenChain volume 211 × all 34 dtypes
- CNN3 TweenChain volume 211 × all 34 dtypes
- ConvT1 TweenChain volume 211 × all 34 dtypes
- ConvT2 TweenChain volume 211 × all 34 dtypes
- ConvT3 TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense TweenChain volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM TweenChain volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

#### `tweensplit_train.go` (80)

- Dense TweenSplit multi-hop Float32 — exact tensors; ROCm loss band (W/loss)
- Dense TweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- RMSNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- LayerNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplit 1-hop Float32 — exact tensors; ROCm loss band (loss)
- Softmax TweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- SwiGLU TweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Sequential TweenSplit 1-hop × all 34 dtypes — spectrum
- Residual TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Residual TweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Mamba TweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Metacognition TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 TweenSplit rank 4 × all 34 dtypes
- CNN3 TweenSplit rank 5 × all 34 dtypes
- ConvT2 TweenSplit rank 4 × all 34 dtypes
- ConvT3 TweenSplit rank 5 × all 34 dtypes
- GDN TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- NestedStack TweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- Embedding TweenSplit 1-hop × all 34 dtypes — spectrum
- MHA TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- MHA TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- CNN1 TweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- ConvT1 TweenSplit 1-hop × all 34 dtypes — spectrum
- RNN TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- RNN TweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplit 1-hop Float32 — exact tensors; ROCm loss band
- LSTM TweenSplit 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenSplit volume 211 × all 34 dtypes
- LayerNorm TweenSplit volume 211 × all 34 dtypes
- SwiGLU TweenSplit volume 211 × all 34 dtypes
- Softmax TweenSplit volume 211 × all 34 dtypes
- Embedding TweenSplit volume 211 × all 34 dtypes
- MHA TweenSplit volume 211 × all 34 dtypes
- RNN TweenSplit volume 211 × all 34 dtypes
- LSTM TweenSplit volume 211 × all 34 dtypes
- Seq TweenSplit volume 211 × all 34 dtypes
- Re TweenSplit volume 211 × all 34 dtypes
- Mb TweenSplit volume 211 × all 34 dtypes
- Mc TweenSplit volume 211 × all 34 dtypes
- Sk TweenSplit volume 211 × all 34 dtypes
- CNN1 TweenSplit volume 211 × all 34 dtypes
- CNN2 TweenSplit volume 211 × all 34 dtypes
- CNN3 TweenSplit volume 211 × all 34 dtypes
- ConvT1 TweenSplit volume 211 × all 34 dtypes
- ConvT2 TweenSplit volume 211 × all 34 dtypes
- ConvT3 TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 1×1×1×1 Float32 — exact tensors; ROCm loss band
- Dense TweenSplit volume 2×1×1×1 chain Float32 — exact tensors; ROCm loss band
- Dense TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RMSNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LayerNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Softmax TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- SwiGLU TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Sequential TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Residual TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Mamba TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Metacognition TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- GDN TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- NestedStack TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- Embedding TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- MHA TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- CNN3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- ConvT3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- RNN TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band
- LSTM TweenSplit volume 2×2×2×1 Float32 — exact tensors; ROCm loss band

### suites/go_v_vulkan (939 leaf cases)

#### `cnn.go` (8)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- CNN1 Float32 Linear pad — exact bits vs Go
- CNN2 Float32 ReLU — exact bits vs Go
- CNN3 Float32 Linear — exact bits vs Go
- CNN1 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings
- CNN2 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings
- CNN3 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `convt.go` (8)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- ConvT1 Float32 Linear pad — exact bits vs Go
- ConvT2 Float32 ReLU — exact bits vs Go
- ConvT3 Float32 Linear out_pad — exact bits vs Go
- ConvT1 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings
- ConvT2 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings
- ConvT3 FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `dense_train.go` (7)

- Dense NormalBP leaf Float32 ReLU — exact tensors; Vulkan loss band (dX/dW/W/loss)
- Dense NormalBP leaf × all 34 dtypes — spectrum + speed
- Dense NormalBP volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense NormalBP volume 1×1×1×2 chain Float32 — exact tensors; Vulkan loss band
- Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Dense NormalBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `embedding.go` (6)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Embedding Float32 Linear [B,T] — exact bits vs Go
- Embedding Float32 Linear [T] rank1 — exact bits vs Go
- Embedding Float32 Linear [B,T,1] rank3 — exact bits vs Go
- Embedding FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `embedding_train.go` (7)

- Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Embedding NormalBP leaf × all 34 dtypes — spectrum + speed
- Embedding NormalBP volume 1×1×1×1 Float32 — exact bits
- Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Embedding NormalBP volume 2×2×2×1 Float32 — exact bits
- Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `gdn.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- GDN Float32 primary — ≥industry vs Go, exact vs C++
- GDN Float32 alt GQA — ≥industry vs Go, exact vs C++
- GDN Float32 mid — parity + Go/C++/Vulkan timings

#### `gdn_train.go` (4)

- GDN NormalBP leaf Float32 — ≥loose vs Go (+ speed)
- GDN NormalBP volume 1×1×1×1 Float32 — ≥industry vs Go
- GDN NormalBP volume 1×1×1×2 chain Float32 — ≥industry vs Go
- GDN NormalBP volume 2×2×2×1 Float32 — ≥industry vs Go

#### `kmeans.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- KMeans probabilities Float32 Linear — exact bits vs Go
- KMeans features Float32 Linear — exact bits vs Go
- KMeans both modes ×34 — exact_bit + Go/C++/Vulkan timings

#### `kmeans_train.go` (8)

- KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)
- KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed
- KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits
- KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits
- KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `layernorm.go` (6)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits vs Go
- LayerNorm Float32 [batch,seq,dim] random affine — exact bits vs Go
- LayerNorm Float32 custom eps — exact bits vs Go
- LayerNorm FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `layernorm_train.go` (7)

- LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)
- LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `lstm.go` (4)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- LSTM Float32 — exact bits vs Go
- LSTM FormatNone ×34 — ≥98% + Go/C++/Vulkan timings

#### `lstm_train.go` (7)

- LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- LSTM NormalBP leaf × all 34 dtypes — spectrum + speed
- LSTM NormalBP volume 1×1×1×1 Float32 — exact bits
- LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits
- LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- LSTM NormalBP volume 2×2×2×1 Float32 — exact bits
- LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mamba.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Mamba Float32 [B,T,D] — exact bits vs Go
- Mamba Float32 alt shape — exact bits vs Go
- Mamba FormatNone ×34 — ≥99% + Go/C++/Vulkan timings

#### `mamba_train.go` (7)

- Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Mamba NormalBP leaf × all 34 dtypes — spectrum + speed
- Mamba NormalBP volume 1×1×1×1 Float32 — exact bits
- Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Mamba NormalBP volume 2×2×2×1 Float32 — exact bits
- Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `meshtween_train.go` (28)

- Dense MeshTween mesh-clock 111 × all 34 dtypes
- Dense MeshTween mesh-clock 211 × all 34 dtypes
- Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `meshtweenchain_train.go` (28)

- Dense MeshTweenChain mesh-clock 111 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 211 × all 34 dtypes
- Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `metacognition.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Metacognition Float32 [B,D] — exact bits vs Go
- Metacognition Float32 [B,T,D] — exact bits vs Go
- Metacognition ×34 — ≥99% + Go/C++/Vulkan timings

#### `metacognition_train.go` (7)

- Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed
- Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits
- Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `mha.go` (6)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- MHA DecoderCausal Float32 — exact bits vs Go
- MHA GQA 4h/2kv Float32 — exact bits vs Go
- MHA Encoder bidirectional Float32 — exact bits vs Go
- MHA DecoderCausal FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `mha_train.go` (8)

- MHA NormalBP leaf DecoderCausal Float32 — exact bits
- MHA NormalBP leaf GQA + Encoder Float32 — exact bits
- MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed
- MHA NormalBP volume 1×1×1×1 Float32 — exact bits
- MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits
- MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- MHA NormalBP volume 2×2×2×1 Float32 — exact bits
- MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `parallel.go` (11)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Parallel Add Float32 [B,D] — exact bits vs Go
- Parallel Add Float32 [B,T,D] — exact bits vs Go
- Parallel Concat Float32 — exact bits vs Go
- Parallel Avg Float32 — exact bits vs Go
- Parallel Filter Float32 — exact bits vs Go
- Parallel Max Float32 — exact bits vs Go
- Parallel SparseK Float32 — exact bits vs Go
- Parallel Disagree Float32 (2+3 branches) — exact bits vs Go
- Parallel Add ×34 — ≥99% + Go/C++/Vulkan timings

#### `parallel_nbp_train.go` (9)

- Dense Add Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP × all 34 dtypes — spectrum
- Dense Avg Parallel NormalBP Float32 — exact bits
- Dense Concat Parallel NormalBP Float32 — exact bits
- Dense Max Parallel NormalBP Float32 — exact bits
- Dense SparseK Parallel NormalBP Float32 — exact bits
- Dense Disagree Parallel NormalBP Float32 — exact bits
- Dense Filter Parallel NormalBP Float32 — exact bits
- Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits

#### `residual.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Residual Dense Float32 [B,D] — exact bits vs Go
- Residual Dense Float32 [B,T,D] — exact bits vs Go
- Residual Dense FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `residual_train.go` (7)

- Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Residual NormalBP leaf × all 34 dtypes — spectrum + speed
- Residual NormalBP volume 1×1×1×1 Float32 — exact bits
- Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Residual NormalBP volume 2×2×2×1 Float32 — exact bits
- Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rmsnorm.go` (6)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- RMSNorm Float32 [batch,dim] γ=1 — exact bits vs Go
- RMSNorm Float32 [batch,seq,dim] random γ — exact bits vs Go
- RMSNorm Float32 custom eps — exact bits vs Go
- RMSNorm FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `rmsnorm_train.go` (7)

- RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)
- RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed
- RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits
- RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `rnn.go` (4)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- RNN Float32 — exact bits vs Go
- RNN FormatNone ×34 — ≥99% + Go/C++/Vulkan timings

#### `rnn_train.go` (7)

- RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- RNN NormalBP leaf × all 34 dtypes — spectrum + speed
- RNN NormalBP volume 1×1×1×1 Float32 — exact bits
- RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits
- RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- RNN NormalBP volume 2×2×2×1 Float32 — exact bits
- RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `seqmix.go` (4)

- Locate welvet.vulkan tool
- seqmix Kind iota + String — Go
- seqmix mixer tags — Go MHA/Mamba/GDN
- seqmix Kind names + tags — Go vs Vulkan WSX1

#### `sequential.go` (5)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Sequential Dense Float32 [B,D] — exact bits vs Go
- Sequential Dense Float32 [B,T,D] — exact bits vs Go
- Sequential Dense FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `sequential_train.go` (7)

- Sequential NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Sequential NormalBP leaf × all 34 dtypes — spectrum + speed
- Sequential NormalBP volume 1×1×1×1 Float32 — exact bits
- Sequential NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Sequential NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Sequential NormalBP volume 2×2×2×1 Float32 — exact bits
- Sequential NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `softmax.go` (11)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- Softmax Standard Float32 — exact bits vs Go
- Softmax Temperature Float32 — exact bits vs Go
- Softmax Masked Float32 — exact bits vs Go
- Softmax Sparse Float32 — exact bits vs Go
- Softmax Entmax Float32 — exact bits vs Go
- Softmax Grid Float32 — exact bits vs Go
- Softmax Hierarchical Float32 — exact bits vs Go
- Softmax Gumbel Float32 — smoke (non-deterministic)
- Softmax all Kind ×34 — exact_bit + Go/C++/Vulkan timings

#### `softmax_train.go` (9)

- Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)
- Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)
- Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed
- Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum
- Softmax NormalBP volume 1×1×1×1 Float32 — exact bits
- Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Softmax NormalBP volume 2×2×2×1 Float32 — exact bits
- Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stack_train.go` (7)

- Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- Stack NormalBP leaf × all 34 dtypes — spectrum + speed
- Stack NormalBP volume 1×1×1×1 Float32 — exact bits
- Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits
- Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- Stack NormalBP volume 2×2×2×1 Float32 — exact bits
- Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `stepbp_train.go` (84)

- Dense StepBP multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense StepBP 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm StepBP 1-hop × all 34 dtypes — spectrum
- LayerNorm StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm StepBP 1-hop × all 34 dtypes — spectrum
- Softmax StepBP 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax StepBP 1-hop × all 34 dtypes — spectrum
- SwiGLU StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU StepBP 1-hop × all 34 dtypes — spectrum
- Sequential StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential StepBP 1-hop × all 34 dtypes — spectrum
- Residual StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- Residual StepBP 1-hop × all 34 dtypes — spectrum
- Mamba StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba StepBP 1-hop × all 34 dtypes — spectrum
- Metacognition StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition StepBP 1-hop × all 34 dtypes — spectrum
- CNN2 StepBP rank 4 × all 34 dtypes
- CNN3 StepBP rank 5 × all 34 dtypes
- ConvT2 StepBP rank 4 × all 34 dtypes
- ConvT3 StepBP rank 5 × all 34 dtypes
- Dense ModeMeshBP volume 211 × all 34 dtypes
- RMSNorm ModeMeshBP volume 211 × all 34 dtypes
- LayerNorm ModeMeshBP volume 211 × all 34 dtypes
- SwiGLU ModeMeshBP volume 211 × all 34 dtypes
- Softmax ModeMeshBP volume 211 × all 34 dtypes
- Embedding ModeMeshBP volume 211 × all 34 dtypes
- MHA ModeMeshBP volume 211 × all 34 dtypes
- RNN ModeMeshBP volume 211 × all 34 dtypes
- LSTM ModeMeshBP volume 211 × all 34 dtypes
- Sequential ModeMeshBP volume 211 × all 34 dtypes
- Residual ModeMeshBP volume 211 × all 34 dtypes
- Mamba ModeMeshBP volume 211 × all 34 dtypes
- Metacognition ModeMeshBP volume 211 × all 34 dtypes
- NestedStack ModeMeshBP volume 211 × all 34 dtypes
- CNN1 ModeMeshBP volume 211 × all 34 dtypes
- CNN2 ModeMeshBP volume 211 × all 34 dtypes
- CNN3 ModeMeshBP volume 211 × all 34 dtypes
- ConvT1 ModeMeshBP volume 211 × all 34 dtypes
- ConvT2 ModeMeshBP volume 211 × all 34 dtypes
- ConvT3 ModeMeshBP volume 211 × all 34 dtypes
- Dense MeshBP mesh-clock 111 × all 34 dtypes
- Dense MeshBP mesh-clock 222 × all 34 dtypes
- Dense MeshBP disabled cell and remote link
- GDN StepBP 1-hop Float32 — ≥strict vs Go (scan FP)
- NestedStack StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepBP 1-hop × all 34 dtypes — spectrum
- Embedding StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding StepBP 1-hop × all 34 dtypes — spectrum
- MHA StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- MHA StepBP 1-hop × all 34 dtypes — spectrum
- CNN1 StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 StepBP 1-hop × all 34 dtypes — spectrum
- ConvT1 StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 StepBP 1-hop × all 34 dtypes — spectrum
- RNN StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- RNN StepBP 1-hop × all 34 dtypes — spectrum
- LSTM StepBP 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM StepBP 1-hop × all 34 dtypes — spectrum
- Dense MeshBP 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshBP 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshBP 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshBP volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `steptween_train.go` (60)

- Dense StepTween multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense StepTween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm StepTween 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm StepTween 1-hop × all 34 dtypes — spectrum
- Softmax StepTween 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax StepTween 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU StepTween 1-hop × all 34 dtypes — spectrum
- Sequential StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential StepTween 1-hop × all 34 dtypes — spectrum
- Residual StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- Residual StepTween 1-hop × all 34 dtypes — spectrum
- Mamba StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba StepTween 1-hop × all 34 dtypes — spectrum
- Embedding StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding StepTween 1-hop × all 34 dtypes — spectrum
- MHA StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- MHA StepTween 1-hop × all 34 dtypes — spectrum
- CNN1 StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 StepTween 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 StepTween 1-hop × all 34 dtypes — spectrum
- RNN StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- RNN StepTween 1-hop × all 34 dtypes — spectrum
- LSTM StepTween 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM StepTween 1-hop × all 34 dtypes — spectrum
- Metacognition StepTween Float32
- Metacognition StepTween × all 34 dtypes
- GDN StepTween Float32 — ≥strict weights; Vulkan loss band
- NestedStack StepTween Float32
- NestedStack StepTween × all 34 dtypes
- Metacognition StepTween volume 222 Float32
- GDN StepTween volume 222 Float32
- NestedStack StepTween volume 222 Float32
- CNN2 StepTween rank 4 × all 34 dtypes
- CNN3 StepTween rank 5 × all 34 dtypes
- ConvT2 StepTween rank 4 × all 34 dtypes
- ConvT3 StepTween rank 5 × all 34 dtypes
- Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `steptweenalt_train.go` (60)

- Dense StepTweenAlt multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense StepTweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax StepTweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Residual StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Residual StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenAlt rank 4 × all 34 dtypes
- CNN3 StepTweenAlt rank 5 × all 34 dtypes
- ConvT2 StepTweenAlt rank 4 × all 34 dtypes
- ConvT3 StepTweenAlt rank 5 × all 34 dtypes
- GDN StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding StepTweenAlt 1-hop × all 34 dtypes — spectrum
- MHA StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- MHA StepTweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 StepTweenAlt 1-hop × all 34 dtypes — spectrum
- RNN StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- RNN StepTweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM StepTweenAlt 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenAlt volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenAlt volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `steptweenchain_train.go` (60)

- Dense StepTweenChain multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum
- Residual StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Residual StepTweenChain 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenChain rank 4 × all 34 dtypes
- CNN3 StepTweenChain rank 5 × all 34 dtypes
- ConvT2 StepTweenChain rank 4 × all 34 dtypes
- ConvT3 StepTweenChain rank 5 × all 34 dtypes
- GDN StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum
- MHA StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- MHA StepTweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum
- RNN StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- RNN StepTweenChain 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `steptweensplit_train.go` (60)

- Dense StepTweenSplit multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense StepTweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax StepTweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Residual StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Residual StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 StepTweenSplit rank 4 × all 34 dtypes
- CNN3 StepTweenSplit rank 5 × all 34 dtypes
- ConvT2 StepTweenSplit rank 4 × all 34 dtypes
- ConvT3 StepTweenSplit rank 5 × all 34 dtypes
- GDN StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding StepTweenSplit 1-hop × all 34 dtypes — spectrum
- MHA StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- MHA StepTweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 StepTweenSplit 1-hop × all 34 dtypes — spectrum
- RNN StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- RNN StepTweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM StepTweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM StepTweenSplit 1-hop × all 34 dtypes — spectrum
- Dense MeshTweenSplit volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenSplit volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM MeshTweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `suite.go` (7)

- Locate welvet.vulkan welvet_dense_fwd
- Locate welvet.cpp welvet_dense_fwd (optional speed)
- Vulkan device identity
- Spectrum ladder self-check
- Dense forward identity+bias Float32 Vulkan — exact bits vs Go
- Dense forward ReLU Float32 Vulkan — exact bits vs Go
- Vulkan FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `swiglu.go` (6)

- Locate welvet.vulkan tool
- Locate welvet.cpp tool (optional speed)
- SwiGLU Float32 [batch,in] — exact bits vs Go
- SwiGLU Float32 [batch,seq,in] — exact bits vs Go
- SwiGLU DefaultFFN Float32 — exact bits vs Go
- SwiGLU FormatNone ×34 — exact_bit + Go/C++/Vulkan timings

#### `swiglu_train.go` (7)

- SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)
- SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed
- SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits
- SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum
- SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits
- SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum

#### `tween_train.go` (80)

- Dense Tween multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense Tween 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm Tween 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm Tween 1-hop × all 34 dtypes — spectrum
- LayerNorm Tween 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm Tween 1-hop × all 34 dtypes — spectrum
- Softmax Tween 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax Tween 1-hop × all 34 dtypes — spectrum
- SwiGLU Tween 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU Tween 1-hop × all 34 dtypes — spectrum
- Sequential Tween 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential Tween 1-hop × all 34 dtypes — spectrum
- Residual Tween 1-hop Float32 — exact tensors; Vulkan loss band
- Residual Tween 1-hop × all 34 dtypes — spectrum
- Mamba Tween 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba Tween 1-hop × all 34 dtypes — spectrum
- Metacognition Tween 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition Tween 1-hop × all 34 dtypes — spectrum
- CNN2 Tween rank 4 × all 34 dtypes
- CNN3 Tween rank 5 × all 34 dtypes
- ConvT2 Tween rank 4 × all 34 dtypes
- ConvT3 Tween rank 5 × all 34 dtypes
- GDN Tween 1-hop Float32 — ≥strict weights; Vulkan loss band
- NestedStack Tween 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack Tween 1-hop × all 34 dtypes — spectrum
- Embedding Tween 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding Tween 1-hop × all 34 dtypes — spectrum
- MHA Tween 1-hop Float32 — exact tensors; Vulkan loss band
- MHA Tween 1-hop × all 34 dtypes — spectrum
- CNN1 Tween 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 Tween 1-hop × all 34 dtypes — spectrum
- ConvT1 Tween 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 Tween 1-hop × all 34 dtypes — spectrum
- RNN Tween 1-hop Float32 — exact tensors; Vulkan loss band
- RNN Tween 1-hop × all 34 dtypes — spectrum
- LSTM Tween 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM Tween 1-hop × all 34 dtypes — spectrum
- RMSNorm Tween volume 211 × all 34 dtypes
- LayerNorm Tween volume 211 × all 34 dtypes
- SwiGLU Tween volume 211 × all 34 dtypes
- Softmax Tween volume 211 × all 34 dtypes
- Embedding Tween volume 211 × all 34 dtypes
- MHA Tween volume 211 × all 34 dtypes
- RNN Tween volume 211 × all 34 dtypes
- LSTM Tween volume 211 × all 34 dtypes
- Seq Tween volume 211 × all 34 dtypes
- Re Tween volume 211 × all 34 dtypes
- Mb Tween volume 211 × all 34 dtypes
- Mc Tween volume 211 × all 34 dtypes
- Sk Tween volume 211 × all 34 dtypes
- CNN1 Tween volume 211 × all 34 dtypes
- CNN2 Tween volume 211 × all 34 dtypes
- CNN3 Tween volume 211 × all 34 dtypes
- ConvT1 Tween volume 211 × all 34 dtypes
- ConvT2 Tween volume 211 × all 34 dtypes
- ConvT3 Tween volume 211 × all 34 dtypes
- Dense Tween volume 211 × all 34 dtypes
- Dense Tween volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense Tween volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM Tween volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `tweenalt_train.go` (80)

- Dense TweenAlt multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense TweenAlt 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenAlt 1-hop × all 34 dtypes — spectrum
- Softmax TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax TweenAlt 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenAlt 1-hop × all 34 dtypes — spectrum
- Sequential TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential TweenAlt 1-hop × all 34 dtypes — spectrum
- Residual TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Residual TweenAlt 1-hop × all 34 dtypes — spectrum
- Mamba TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba TweenAlt 1-hop × all 34 dtypes — spectrum
- Metacognition TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN2 TweenAlt rank 4 × all 34 dtypes
- CNN3 TweenAlt rank 5 × all 34 dtypes
- ConvT2 TweenAlt rank 4 × all 34 dtypes
- ConvT3 TweenAlt rank 5 × all 34 dtypes
- GDN TweenAlt 1-hop Float32 — ≥strict weights; finite GDN loss band
- NestedStack TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack TweenAlt 1-hop × all 34 dtypes — spectrum
- Embedding TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding TweenAlt 1-hop × all 34 dtypes — spectrum
- MHA TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- MHA TweenAlt 1-hop × all 34 dtypes — spectrum
- CNN1 TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 TweenAlt 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenAlt 1-hop × all 34 dtypes — spectrum
- RNN TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- RNN TweenAlt 1-hop × all 34 dtypes — spectrum
- LSTM TweenAlt 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM TweenAlt 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenAlt volume 211 × all 34 dtypes
- LayerNorm TweenAlt volume 211 × all 34 dtypes
- SwiGLU TweenAlt volume 211 × all 34 dtypes
- Softmax TweenAlt volume 211 × all 34 dtypes
- Embedding TweenAlt volume 211 × all 34 dtypes
- MHA TweenAlt volume 211 × all 34 dtypes
- RNN TweenAlt volume 211 × all 34 dtypes
- LSTM TweenAlt volume 211 × all 34 dtypes
- Seq TweenAlt volume 211 × all 34 dtypes
- Re TweenAlt volume 211 × all 34 dtypes
- Mb TweenAlt volume 211 × all 34 dtypes
- Mc TweenAlt volume 211 × all 34 dtypes
- Sk TweenAlt volume 211 × all 34 dtypes
- CNN1 TweenAlt volume 211 × all 34 dtypes
- CNN2 TweenAlt volume 211 × all 34 dtypes
- CNN3 TweenAlt volume 211 × all 34 dtypes
- ConvT1 TweenAlt volume 211 × all 34 dtypes
- ConvT2 TweenAlt volume 211 × all 34 dtypes
- ConvT3 TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 211 × all 34 dtypes
- Dense TweenAlt volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense TweenAlt volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM TweenAlt volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `tweenchain_train.go` (80)

- Dense TweenChain multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense TweenChain 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenChain 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenChain 1-hop × all 34 dtypes — spectrum
- Softmax TweenChain 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax TweenChain 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenChain 1-hop × all 34 dtypes — spectrum
- Sequential TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential TweenChain 1-hop × all 34 dtypes — spectrum
- Residual TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Residual TweenChain 1-hop × all 34 dtypes — spectrum
- Mamba TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba TweenChain 1-hop × all 34 dtypes — spectrum
- Metacognition TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition TweenChain 1-hop × all 34 dtypes — spectrum
- CNN2 TweenChain rank 4 × all 34 dtypes
- CNN3 TweenChain rank 5 × all 34 dtypes
- ConvT2 TweenChain rank 4 × all 34 dtypes
- ConvT3 TweenChain rank 5 × all 34 dtypes
- GDN TweenChain 1-hop Float32 — ≥strict weights; Vulkan loss band
- NestedStack TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack TweenChain 1-hop × all 34 dtypes — spectrum
- Embedding TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding TweenChain 1-hop × all 34 dtypes — spectrum
- MHA TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- MHA TweenChain 1-hop × all 34 dtypes — spectrum
- CNN1 TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 TweenChain 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenChain 1-hop × all 34 dtypes — spectrum
- RNN TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- RNN TweenChain 1-hop × all 34 dtypes — spectrum
- LSTM TweenChain 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM TweenChain 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenChain volume 211 × all 34 dtypes
- LayerNorm TweenChain volume 211 × all 34 dtypes
- SwiGLU TweenChain volume 211 × all 34 dtypes
- Softmax TweenChain volume 211 × all 34 dtypes
- Embedding TweenChain volume 211 × all 34 dtypes
- MHA TweenChain volume 211 × all 34 dtypes
- RNN TweenChain volume 211 × all 34 dtypes
- LSTM TweenChain volume 211 × all 34 dtypes
- Seq TweenChain volume 211 × all 34 dtypes
- Re TweenChain volume 211 × all 34 dtypes
- Mb TweenChain volume 211 × all 34 dtypes
- Mc TweenChain volume 211 × all 34 dtypes
- Sk TweenChain volume 211 × all 34 dtypes
- CNN1 TweenChain volume 211 × all 34 dtypes
- CNN2 TweenChain volume 211 × all 34 dtypes
- CNN3 TweenChain volume 211 × all 34 dtypes
- ConvT1 TweenChain volume 211 × all 34 dtypes
- ConvT2 TweenChain volume 211 × all 34 dtypes
- ConvT3 TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 211 × all 34 dtypes
- Dense TweenChain volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense TweenChain volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM TweenChain volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

#### `tweensplit_train.go` (80)

- Dense TweenSplit multi-hop Float32 — exact tensors; Vulkan loss band (W/loss)
- Dense TweenSplit 1-hop × all 34 dtypes — spectrum + speed
- RMSNorm TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- LayerNorm TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenSplit 1-hop × all 34 dtypes — spectrum
- Softmax TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band (loss)
- Softmax TweenSplit 1-hop × all 34 dtypes — spectrum
- SwiGLU TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenSplit 1-hop × all 34 dtypes — spectrum
- Sequential TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Sequential TweenSplit 1-hop × all 34 dtypes — spectrum
- Residual TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Residual TweenSplit 1-hop × all 34 dtypes — spectrum
- Mamba TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Mamba TweenSplit 1-hop × all 34 dtypes — spectrum
- Metacognition TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Metacognition TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN2 TweenSplit rank 4 × all 34 dtypes
- CNN3 TweenSplit rank 5 × all 34 dtypes
- ConvT2 TweenSplit rank 4 × all 34 dtypes
- ConvT3 TweenSplit rank 5 × all 34 dtypes
- GDN TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- NestedStack TweenSplit 1-hop × all 34 dtypes — spectrum
- Embedding TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- Embedding TweenSplit 1-hop × all 34 dtypes — spectrum
- MHA TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- MHA TweenSplit 1-hop × all 34 dtypes — spectrum
- CNN1 TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- CNN1 TweenSplit 1-hop × all 34 dtypes — spectrum
- ConvT1 TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenSplit 1-hop × all 34 dtypes — spectrum
- RNN TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- RNN TweenSplit 1-hop × all 34 dtypes — spectrum
- LSTM TweenSplit 1-hop Float32 — exact tensors; Vulkan loss band
- LSTM TweenSplit 1-hop × all 34 dtypes — spectrum
- RMSNorm TweenSplit volume 211 × all 34 dtypes
- LayerNorm TweenSplit volume 211 × all 34 dtypes
- SwiGLU TweenSplit volume 211 × all 34 dtypes
- Softmax TweenSplit volume 211 × all 34 dtypes
- Embedding TweenSplit volume 211 × all 34 dtypes
- MHA TweenSplit volume 211 × all 34 dtypes
- RNN TweenSplit volume 211 × all 34 dtypes
- LSTM TweenSplit volume 211 × all 34 dtypes
- Seq TweenSplit volume 211 × all 34 dtypes
- Re TweenSplit volume 211 × all 34 dtypes
- Mb TweenSplit volume 211 × all 34 dtypes
- Mc TweenSplit volume 211 × all 34 dtypes
- Sk TweenSplit volume 211 × all 34 dtypes
- CNN1 TweenSplit volume 211 × all 34 dtypes
- CNN2 TweenSplit volume 211 × all 34 dtypes
- CNN3 TweenSplit volume 211 × all 34 dtypes
- ConvT1 TweenSplit volume 211 × all 34 dtypes
- ConvT2 TweenSplit volume 211 × all 34 dtypes
- ConvT3 TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 211 × all 34 dtypes
- Dense TweenSplit volume 1×1×1×1 Float32 — exact tensors; Vulkan loss band
- Dense TweenSplit volume 2×1×1×1 chain Float32 — exact tensors; Vulkan loss band
- Dense TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RMSNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LayerNorm TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Softmax TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- SwiGLU TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Sequential TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Residual TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Mamba TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Metacognition TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- GDN TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- NestedStack TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- Embedding TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- MHA TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- CNN3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT1 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT2 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- ConvT3 TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- RNN TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band
- LSTM TweenSplit volume 2×2×2×1 Float32 — exact tensors; Vulkan loss band

## Go `func Test*` entrypoints

### tests/cnn1 (1)

- `TestCNN1Suite`

### tests/cnn2 (1)

- `TestCNN2Suite`

### tests/cnn3 (1)

- `TestCNN3Suite`

### tests/convt1 (1)

- `TestConvT1Suite`

### tests/convt2 (1)

- `TestConvT2Suite`

### tests/convt3 (1)

- `TestConvT3Suite`

### tests/dense (3)

- `TestDenseSuite`
- `TestGemvQ4ParallelSmoke`
- `TestLinearGradInAndGradWOnly`

### tests/densenet (2)

- `TestVolumetricDenseChain`
- `TestVolumetricRemoteHop`

### tests/dna (3)

- `TestExtractCompareImmutable`
- `TestDNAAcrossDTypeQuant`
- `TestDNAMultiLayerOps`

### tests/embedding (1)

- `TestEmbeddingSuite`

### tests/entity (9)

- `TestCameralBicameralRoundTrip`
- `TestCameralBranchModesPersist`
- `TestCameralTrainAfterLoad`
- `TestCameralMetaSandwich`
- `TestCameralWrappedCNN`
- `TestParseTrainModeAliases`
- `TestLoadCameralRejectsTransformer`
- `TestWriteInspectLoadBlobRoundTrip`
- `TestImportFromHFRejectsMissing`

### tests/evolution (2)

- `TestSpliceBlend`
- `TestCloneSpliceMultiLayer`

### tests/fusedgpu (3)

- `TestClampAttnMaxSeq`
- `TestClampMaxSeqForKVBudget`
- `TestEstimateHybridKVBytes`

### tests/gdn (1)

- `TestGDNSuite`

### tests/go_v_cpp (72)

- `TestGoVCpp`
- `TestGoVCppMHA`
- `TestGoVCppSwiGLU`
- `TestGoVCppRMSNorm`
- `TestGoVCppLayerNorm`
- `TestGoVCppCNN`
- `TestGoVCppConvT`
- `TestGoVCppRNN`
- `TestGoVCppLSTM`
- `TestGoVCppEmbedding`
- `TestGoVCppSoftmax`
- `TestGoVCppKMeans`
- `TestGoVCppSequential`
- `TestGoVCppResidual`
- `TestGoVCppMamba`
- `TestGoVCppMetacognition`
- `TestGoVCppGDN`
- `TestGoVCppSeqmix`
- `TestGoVCppParallel`
- `TestGoVCppDenseTrain`
- `TestGoVCppRMSNormTrain`
- `TestGoVCppLayerNormTrain`
- `TestGoVCppSwiGLUTrain`
- `TestGoVCppSoftmaxTrain`
- `TestGoVCppEmbeddingTrain`
- `TestGoVCppMHATrain`
- `TestGoVCppCNNTrain`
- `TestGoVCppConvTTrain`
- `TestGoVCppLSTMTrain`
- `TestGoVCppRNNTrain`
- `TestGoVCppSequentialTrain`
- `TestGoVCppResidualTrain`
- `TestGoVCppMambaTrain`
- `TestGoVCppStepBPTrain`
- `TestGoVCppTweenTrain`
- `TestGoVCppTweenChainTrain`
- `TestGoVCppStepTweenTrain`
- `TestGoVCppStepTweenChainTrain`
- `TestGoVCppMeshTweenTrain`
- `TestGoVCppMeshTweenChainTrain`
- `TestGoVCppTweenSplitTrain`
- `TestGoVCppStepTweenSplitTrain`
- `TestGoVCppTweenAltTrain`
- `TestGoVCppStepTweenAltTrain`
- `TestGoVCppTweenSplitHeadProxyTrain`
- `TestGoVCppTweenSplitLinearTrain`
- `TestGoVCppTweenSplitFastProxyTrain`
- `TestGoVCppTweenSplitLinearCacheTrain`
- `TestGoVCppTweenSplitHeadProxyAsyncTrain`
- `TestGoVCppTweenSplitSparseTrain`
- `TestGoVCppMeshTweenSplitTrain`
- `TestGoVCppMeshTweenAltTrain`
- `TestGoVCppMeshTweenSplitFastProxyTrain`
- `TestGoVCppMeshTweenSplitSparseTrain`
- `TestGoVCppStepTweenSplitHeadProxyTrain`
- `TestGoVCppStepTweenSplitLinearTrain`
- `TestGoVCppStepTweenSplitFastProxyTrain`
- `TestGoVCppStepTweenSplitLinearCacheTrain`
- `TestGoVCppStepTweenSplitHeadProxyAsyncTrain`
- `TestGoVCppStepTweenSplitSparseTrain`
- `TestGoVCppFreezeTrain`
- `TestGoVCppAdversarialTrain`
- `TestGoVCppMemoryTrain`
- `TestGoVCppShadowTrain`
- `TestGoVCppMetacognitionTrain`
- `TestGoVCppGDNTrain`
- `TestGoVCppStackTrain`
- `TestGoVCppKMeansTrain`
- `TestGoVCppParallelNBPTrain`
- `TestGoVCppParallelTweenTrain`
- `TestGoVCppParallelPolyTrain`
- `TestGoVCppAllTrain`

### tests/go_v_cuda (79)

- `TestGoVCudaDenseForward`
- `TestGoVCudaRMSNorm`
- `TestGoVCudaLayerNorm`
- `TestGoVCudaSoftmax`
- `TestGoVCudaSwiGLU`
- `TestGoVCudaEmbedding`
- `TestGoVCudaMHA`
- `TestGoVCudaCNN`
- `TestGoVCudaConvT`
- `TestGoVCudaRNN`
- `TestGoVCudaLSTM`
- `TestGoVCudaSequential`
- `TestGoVCudaResidual`
- `TestGoVCudaMamba`
- `TestGoVCudaMetacognition`
- `TestGoVCudaGDN`
- `TestGoVCudaKMeans`
- `TestGoVCudaParallel`
- `TestGoVCudaSeqmix`
- `TestGoVCudaDenseTrain`
- `TestGoVCudaRMSNormTrain`
- `TestGoVCudaLayerNormTrain`
- `TestGoVCudaSoftmaxTrain`
- `TestGoVCudaSwiGLUTrain`
- `TestGoVCudaEmbeddingTrain`
- `TestGoVCudaMHATrain`
- `TestGoVCudaCNNTrain`
- `TestGoVCudaConvTTrain`
- `TestGoVCudaRNNTrain`
- `TestGoVCudaLSTMTrain`
- `TestGoVCudaSequentialTrain`
- `TestGoVCudaResidualTrain`
- `TestGoVCudaStackTrain`
- `TestGoVCudaMambaTrain`
- `TestGoVCudaMetacognitionTrain`
- `TestGoVCudaGDNTrain`
- `TestGoVCudaNormalBP`
- `TestGoVCudaKMeansTrain`
- `TestGoVCudaParallelTrain`
- `TestGoVCudaParallelPolyTrain`
- `TestGoVCudaNormalBPEdges`
- `TestGoVCudaStepBPTrain`
- `TestGoVCudaMeshBPTrain`
- `TestGoVCudaTweenTrain`
- `TestGoVCudaTweenChainTrain`
- `TestGoVCudaTweenSplitTrain`
- `TestGoVCudaStepTweenSplitTrain`
- `TestGoVCudaTweenAltTrain`
- `TestGoVCudaStepTweenAltTrain`
- `TestGoVCudaMeshTweenTrain`
- `TestGoVCudaMeshTweenChainTrain`
- `TestGoVCudaStepTweenTrain`
- `TestGoVCudaStepTweenChainTrain`
- `TestGoVCudaProxyTrain`
- `TestGoVCudaProxyVolumeTrain`
- `TestGoVCudaMeshSplitModes`
- `TestGoVCudaStepProxyTrain`
- `TestGoVCudaCameralTrain`
- `TestGoVCudaCamSync`
- `TestGoNativeCudaSaveReload`
- `TestGoNativeCudaErrorsAndConcurrency`
- `TestGoNativeCudaCamSync`
- `TestGoNativeCudaUnsupportedGridMode`
- `TestGoNativeCudaIndependentLink`
- `TestGoNativeCudaRoutingCPPRequired`
- `TestGoNativeCudaRoutingStack`
- `TestGoNativeCudaRoutingGrid`
- `TestGoNativeCudaRoutingGuards`
- `TestGoNativeCudaRoutingMixedModes`
- `TestGoNativeCudaRoutingOverrides`
- `TestGoNativeCudaRoutingPendingCPP`
- `TestGoNativeCudaRoutingBadLibrary`
- `TestGoNativeCudaRoutingNestedOverrides`
- `TestGoNativeCudaRoutingCredit`
- `TestGoNativeCudaRoutingParallelGate`
- `TestGoNativeCudaIntegration`
- `TestGoNativeCudaMesh`
- `TestGoNativeCudaRoutingLayers`
- `TestGoNativeCudaRoutingModes`

### tests/go_v_metal (87)

- `TestAppleMambaStress`
- `TestAppleNumberedExamples`
- `TestGoVMetalCameralTrain`
- `TestGoVMetalCamSync`
- `TestGoVMetalCNN`
- `TestGoVMetalCNNTrain`
- `TestGoVMetalConvT`
- `TestGoVMetalConvTTrain`
- `TestGoVMetalDenseTrain`
- `TestGoVMetalEmbedding`
- `TestGoVMetalEmbeddingTrain`
- `TestGoVMetalGDN`
- `TestGoVMetalGDNTrain`
- `TestGoReleaseScriptMarkdown`
- `TestGoVMetalDenseForward`
- `TestGoVMetalKMeans`
- `TestGoVMetalLayerNorm`
- `TestGoVMetalLayerNormTrain`
- `TestGoVMetalLineAltEdge`
- `TestGoVMetalLSTM`
- `TestGoVMetalLSTMTrain`
- `TestGoVMetalMamba`
- `TestGoVMetalMambaTrain`
- `TestGoVMetalMeshSplitModes`
- `TestGoVMetalMeshBPTrain`
- `TestGoVMetalMeshTweenTrain`
- `TestGoVMetalMeshTweenChainTrain`
- `TestGoVMetalMetacognition`
- `TestGoVMetalMetacognitionTrain`
- `TestGoVMetalOnly`
- `TestMetalReleaseScript`
- `TestGoVMetalMHA`
- `TestGoVMetalMHATrain`
- `TestGoNativeCPPRouting`
- `TestGoNativeMetalSaveReload`
- `TestGoNativeMetalErrorsAndConcurrency`
- `TestGoNativeMetalCamSync`
- `TestGoNativeMetalUnsupportedGridMode`
- `TestGoNativeMetalIndependentLink`
- `TestGoNativeMetalRoutingCPPRequired`
- `TestGoNativeMetalRoutingStack`
- `TestGoNativeMetalRoutingGrid`
- `TestGoNativeMetalRoutingGuards`
- `TestGoNativeMetalRoutingMixedModes`
- `TestGoNativeMetalRoutingOverrides`
- `TestGoNativeMetalRoutingPendingCPP`
- `TestGoNativeMetalRoutingBadLibrary`
- `TestGoNativeMetalRoutingNestedOverrides`
- `TestGoNativeMetalRoutingCredit`
- `TestGoNativeMetalRoutingParallelGate`
- `TestGoNativeMetalIntegration`
- `TestGoNativeMetalMesh`
- `TestGoNativeMetalRoutingLayers`
- `TestGoNativeMetalRoutingModes`
- `TestGoVMetalNormalBP`
- `TestGoVMetalParallel`
- `TestGoVMetalProxyTrain`
- `TestGoVMetalProxyVolumeTrain`
- `TestGoVMetalResidual`
- `TestGoVMetalResidualTrain`
- `TestGoVMetalRMSNorm`
- `TestGoVMetalRMSNormTrain`
- `TestGoVMetalRNN`
- `TestGoVMetalRNNTrain`
- `TestGoVMetalSeqmix`
- `TestGoVMetalSequential`
- `TestGoVMetalSequentialTrain`
- `TestGoVMetalSoftmax`
- `TestGoVMetalSoftmaxTrain`
- `TestGoVMetalStackTrain`
- `TestGoVMetalStepProxyTrain`
- `TestGoVMetalStepBPTrain`
- `TestGoVMetalStepTweenTrain`
- `TestGoVMetalStepTweenAltTrain`
- `TestGoVMetalStepTweenChainTrain`
- `TestGoVMetalStepTweenSplitTrain`
- `TestGoStoreReplaceInvalidatesWires`
- `TestGoVMetalSwiGLU`
- `TestGoVMetalSwiGLUTrain`
- `TestGoVMetalTweenTrain`
- `TestGoVMetalTweenAltTrain`
- `TestGoVMetalTweenChainTrain`
- `TestGoVMetalTweenSplitTrain`
- `TestGoVMetalVolume`
- `TestGoVMetalVolumeMHA`
- `TestGoVMetalVolumeSwiGLU`
- `TestGoVMetalVolumeAllLayers`

### tests/go_v_mlx (4)

- `TestGoVMLX`
- `TestGoVMLXAll`
- `TestGoVMLXCaps`
- `TestGoVMLXDense`

### tests/go_v_qnn (2)

- `TestGoVQNN`
- `TestGoVQNNMatrix`

### tests/go_v_rocm (61)

- `TestGoVRocmCameralTrain`
- `TestGoVRocmCamSync`
- `TestGoVRocmCNN`
- `TestGoVRocmCNNTrain`
- `TestGoVRocmConvT`
- `TestGoVRocmConvTTrain`
- `TestGoVRocmDenseTrain`
- `TestGoVRocmEmbedding`
- `TestGoVRocmEmbeddingTrain`
- `TestGoVRocmGDN`
- `TestGoVRocmGDNTrain`
- `TestGoVRocmDenseForward`
- `TestGoVRocmKMeans`
- `TestGoVRocmKMeansTrain`
- `TestGoVRocmLayerNorm`
- `TestGoVRocmLayerNormTrain`
- `TestGoVRocmLSTM`
- `TestGoVRocmLSTMTrain`
- `TestGoVRocmMamba`
- `TestGoVRocmMambaTrain`
- `TestGoVRocmMeshSplitModes`
- `TestGoVRocmMeshBPTrain`
- `TestGoVRocmMeshTweenTrain`
- `TestGoVRocmMeshTweenChainTrain`
- `TestGoVRocmMetacognition`
- `TestGoVRocmMetacognitionTrain`
- `TestGoVRocmMHA`
- `TestGoVRocmMHATrain`
- `TestGoVRocmNormalBP`
- `TestGoVRocmParallel`
- `TestGoVRocmParallelTrain`
- `TestGoVRocmParallelPolyTrain`
- `TestGoVRocmNormalBPEdges`
- `TestGoVRocmProxyTrain`
- `TestGoVRocmProxyVolumeTrain`
- `TestGoVRocmResidual`
- `TestGoVRocmResidualTrain`
- `TestGoVRocmRMSNorm`
- `TestGoVRocmRMSNormTrain`
- `TestGoVRocmRNN`
- `TestGoVRocmRNNTrain`
- `TestGoVRocmSeqmix`
- `TestGoVRocmSequential`
- `TestGoVRocmSequentialTrain`
- `TestGoVRocmSmokeVersion`
- `TestGoVRocmSmokeGaps`
- `TestGoVRocmSoftmax`
- `TestGoVRocmSoftmaxTrain`
- `TestGoVRocmStackTrain`
- `TestGoVRocmStepProxyTrain`
- `TestGoVRocmStepBPTrain`
- `TestGoVRocmStepTweenTrain`
- `TestGoVRocmStepTweenAltTrain`
- `TestGoVRocmStepTweenChainTrain`
- `TestGoVRocmStepTweenSplitTrain`
- `TestGoVRocmSwiGLU`
- `TestGoVRocmSwiGLUTrain`
- `TestGoVRocmTweenTrain`
- `TestGoVRocmTweenAltTrain`
- `TestGoVRocmTweenChainTrain`
- `TestGoVRocmTweenSplitTrain`

### tests/go_v_vulkan (61)

- `TestGoVVulkanCameralTrain`
- `TestGoVVulkanCamSync`
- `TestGoVVulkanCNN`
- `TestGoVVulkanCNNTrain`
- `TestGoVVulkanConvT`
- `TestGoVVulkanConvTTrain`
- `TestGoVVulkanDenseTrain`
- `TestGoVVulkanEmbedding`
- `TestGoVVulkanEmbeddingTrain`
- `TestGoVVulkanGDN`
- `TestGoVVulkanGDNTrain`
- `TestGoVVulkanKMeans`
- `TestGoVVulkanKMeansTrain`
- `TestGoVVulkanLayerNorm`
- `TestGoVVulkanLayerNormTrain`
- `TestGoVVulkanLSTM`
- `TestGoVVulkanLSTMTrain`
- `TestGoVVulkanMamba`
- `TestGoVVulkanMambaTrain`
- `TestGoVVulkanMeshSplitModes`
- `TestGoVVulkanMeshBPTrain`
- `TestGoVVulkanMeshTweenTrain`
- `TestGoVVulkanMeshTweenChainTrain`
- `TestGoVVulkanMetacognition`
- `TestGoVVulkanMetacognitionTrain`
- `TestGoVVulkanMHA`
- `TestGoVVulkanMHATrain`
- `TestGoVVulkanNormalBP`
- `TestGoVVulkanParallel`
- `TestGoVVulkanParallelTrain`
- `TestGoVVulkanParallelPolyTrain`
- `TestGoVVulkanNormalBPEdges`
- `TestGoVVulkanProxyTrain`
- `TestGoVVulkanProxyVolumeTrain`
- `TestGoVVulkanResidual`
- `TestGoVVulkanResidualTrain`
- `TestGoVVulkanRMSNorm`
- `TestGoVVulkanRMSNormTrain`
- `TestGoVVulkanRNN`
- `TestGoVVulkanRNNTrain`
- `TestGoVVulkanSeqmix`
- `TestGoVVulkanSequential`
- `TestGoVVulkanSequentialTrain`
- `TestGoVVulkanSoftmax`
- `TestGoVVulkanSoftmaxTrain`
- `TestGoVVulkanStackTrain`
- `TestGoVVulkanStepProxyTrain`
- `TestGoVVulkanStepBPTrain`
- `TestGoVVulkanStepTweenTrain`
- `TestGoVVulkanStepTweenAltTrain`
- `TestGoVVulkanStepTweenChainTrain`
- `TestGoVVulkanStepTweenSplitTrain`
- `TestGoVVulkanSwiGLU`
- `TestGoVVulkanSwiGLUTrain`
- `TestGoVVulkanTweenTrain`
- `TestGoVVulkanTweenAltTrain`
- `TestGoVVulkanTweenChainTrain`
- `TestGoVVulkanTweenSplitTrain`
- `TestGoVVulkanDenseForward`
- `TestGoVVulkanSmokeVersion`
- `TestGoVVulkanSmokeGaps`

### tests/hf (3)

- `TestDetectArchitectureLlamaStyle`
- `TestInspectSnapshot`
- `TestMaxSeqLenFromConfig`

### tests/kmeans (1)

- `TestKMeansSuite`

### tests/layernorm (1)

- `TestLayerNormSuite`

### tests/lstm (1)

- `TestLSTMSuite`

### tests/lucy (3)

- `TestLucyScoreEquation`
- `TestSoftAccOne`
- `TestSoftAccProb`

### tests/mamba (1)

- `TestMambaSuite`

### tests/metacognition (2)

- `TestMetacognitionSuite`
- `TestBackwardNilPreDoesNotPanic`

### tests/mha (1)

- `TestMHASuite`

### tests/parallel (33)

- `TestAllCreditModesMoveBicameral`
- `TestAllStackLocalModesMoveBicameral`
- `TestStepAndNonStepSplitBothTrain`
- `TestOpenSplitTapeAllSplitModes`
- `TestTrainMSEParallelRootCredit`
- `TestCombineModesWithFastProxy`
- `TestHemiNCreditModes`
- `TestMixedFastProxyAndStepBPTrains`
- `TestSequentialAndResidualMidCredit`
- `TestSparseRotatesHiddenLeaves`
- `TestParseTrainModeRoundTripAll`
- `TestRequiresGridOnlyMesh`
- `TestWrappedCNNAllCreditModes`
- `TestInheritResolvesToNormalBP`
- `TestSuiteCameralCreditModes`
- `TestSuiteCameralCreditTrainGrids`
- `TestAllMeshCreditModesMoveBicameral`
- `TestMeshCreditVariantsDispatch`
- `TestParseMeshCreditAliases`
- `TestMixedSequentialResidualCredit`
- `TestResidualGraftParallel`
- `TestCameralModesIncludeCreditAndMesh`
- `TestTweenSplitUpdatesWholeSandwich`
- `TestTweenSplitWeightedModesMove`
- `TestSplitTapeScoresThenTrains`
- `TestLinearCacheAndAsyncSecondStep`
- `TestSplitFamilyModes`
- `TestTweenAltUpdatesWholeSandwich`
- `TestTweenAltFamily`
- `TestLinearMetacognitionSandwichNoPanic`
- `TestTweenTrainsWrappedCNN`
- `TestParallelSuite`
- `Test49AllTrainModesCubes`

### tests/quant (1)

- `TestMatVecBinaryG128Exact`

### tests/residual (2)

- `TestNewFromOpsRMSSwiGLU`
- `TestResidualSuite`

### tests/rmsnorm (1)

- `TestRMSNormSuite`

### tests/rnn (1)

- `TestRNNSuite`

### tests/sampling (4)

- `TestArgMaxAndSampleTopKGreedy`
- `TestSampleTopKRestricted`
- `TestBanAndPenalty`
- `TestReplyLooksDegenerate`

### tests/sequential (2)

- `TestNewFromOpsDenseRMSSwiGLU`
- `TestSequentialSuite`

### tests/serialization (1)

- `TestSerializationSuite`

### tests/seven (1)

- `TestSevenSuite`

### tests/simd (1)

- `TestSaxpyF32`

### tests/softmax (1)

- `TestSoftmaxSuite`

### tests/step (2)

- `TestStepSuite`
- `TestStepMeshSmoke`

### tests/swiglu (1)

- `TestSwiGLUSuite`

### tests/tanhi (1)

- `TestEmitDisabledNoPanic`

### tests/telemetry (1)

- `TestBlueprintDenseParams`

### tests/transformer (2)

- `TestGenOptionsSampleTopKDefaults`
- `TestApplyPartialRoPERotateHalf`

### tests/tween (2)

- `TestTweenMultiLayerChainRule`
- `TestTweenGapReduce`

### tests/wav2vec2 (3)

- `TestCTCVocabDecode`
- `TestTranscribeSample3s`
- `TestEntityRoundTrip`

### tests/weights (1)

- `TestWeightsSuite`
