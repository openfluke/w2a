# go_v_qnn results — layer × dtype (Hexagon V73 HTP)

**Host:** Windows ARM64 · Snapdragon X · `nativeqnn` + QAIRT 2.42  
**Run:** 2026-10-23 · **large** matrix shapes (same geometry Go / native / NPU)  
**Shapes:** dense 32×512×512 · softmax 64×4096 · norms [8,128,512] · cnn2 4×64×64×32 · cnn3 2×16³×16 · convt2 4×32×32×32 · convt3 1×12³×16 · lstm T8 B4 64/64 · emb 4k×256 n=512 · residual 128k · swiglu 8×256×768 · mha 2×32×128×4

### Timing (read this)

- **NPU DLL exec** — `mono_ns` *inside* `welvet.qnn` around `graphExecute` only (excludes cgo, f32↔dtype pack/unpack, bias epilogue).
- **Go→NPU call** — Go wall around full C ABI (cgo + pack + DLL + unpack). Online layers also rebuild the graph each call.
- **Dense** = prepare-once + warmed execute. Other layers = online compose today.
- **Go / native** = **f32 CPU only** (one column). Welvet host does not run HTP quant kernels — dtype matrix is **NPU-only**. Compare each NPU dtype row against that f32 host baseline.

Legend: ✅ pass · ❌ fail · — caps skip (incl. bf16 = V81+)

## Status (layer × dtype)

| Layer | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |
|-------|-----|-----|----|----|-----|-----|------|
| dense | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| softmax | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| rmsnorm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| layernorm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| cnn2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| cnn3 | ✅ | ✅ | ✅ | ✅ | — | — | — |
| convt2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| convt3 | ✅ | ✅ | ✅ | ✅ | — | — | — |
| lstm | ❌ | ❌ | — | — | — | — | — |
| embedding | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| residual | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| swiglu | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| mha | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |

## Go / native exec (f32 CPU, same shapes as NPU)

One number per layer — host has no per-dtype HTP path. `—` = not wired yet.

| Layer | Go f32 | Native f32 |
|-------|--------|------------|
| dense | 3.05ms | 9.64ms |
| softmax | 4.31ms | 2.72ms |
| rmsnorm | 9.40ms | 2.43ms |
| layernorm | 6.82ms | 2.60ms |
| cnn2 | 12.56ms | — |
| cnn3 | 1.33ms | — |
| convt2 | 3.14ms | — |
| convt3 | 285µs | — |
| lstm | 1.38ms | — |
| embedding | 69µs | 82µs |
| residual | 91µs | — |
| swiglu | 1.92ms | 5.49ms |
| mha | 4.51ms | 5.49ms |

## NPU DLL exec (layer × dtype)

`graphExecute` only, measured in `welvet.qnn`. **This** is where dtypes differ.

| Layer | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |
|-------|-----|-----|----|----|-----|-----|------|
| dense | 201µs | 194µs | 183µs | 164µs | 164µs | 249µs | — |
| softmax | 656µs | 446µs | 485µs | 453µs | 785µs | 530µs | — |
| rmsnorm | 2.1ms | 1.7ms | 1.0ms | 897µs | 1.8ms | 1.3ms | — |
| layernorm | 1.1ms | 553µs | 594µs | 635µs | 650µs | 587µs | — |
| cnn2 | 1.1ms | 833µs | 763µs | 1.0ms | 1.8ms | 1.1ms | — |
| cnn3 | 573µs | 509µs | 488µs | 647µs | — | — | — |
| convt2 | 724µs | 542µs | 603µs | 620µs | 877µs | 506µs | — |
| convt3 | 499µs | 510µs | 424µs | 378µs | — | — | — |
| lstm | FAIL | FAIL | — | — | — | — | — |
| embedding | 1.9ms | 572µs | 435µs | 464µs | 566µs | 524µs | — |
| residual | 1.6ms | 6.2ms | 470µs | 417µs | 438µs | 434µs | — |
| swiglu | 589µs | 378µs | 666µs | 362µs | 386µs | 352µs | — |
| mha | 439µs | 372µs | 382µs | 386µs | 530µs | 413µs | — |

## Go→NPU call (layer × dtype)

Full ABI from Go. Includes create/finalize for online layers.

| Layer | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |
|-------|-----|-----|----|----|-----|-----|------|
| dense | 369µs | 194µs | 578µs | 164µs | 503µs | 521µs | — |
| softmax | 30.0ms | 10.5ms | 16.5ms | 18.4ms | 25.5ms | 8.0ms | — |
| rmsnorm | 41.2ms | 33.5ms | 21.6ms | 13.8ms | 42.0ms | 30.7ms | — |
| layernorm | 22.2ms | 13.7ms | 20.8ms | 10.6ms | 13.6ms | 20.7ms | — |
| cnn2 | 32.4ms | 22.8ms | 37.8ms | 24.0ms | 22.5ms | 31.3ms | — |
| cnn3 | 42.9ms | 28.8ms | 45.5ms | 35.7ms | — | — | — |
| convt2 | 18.7ms | 28.9ms | 16.8ms | 16.8ms | 25.4ms | 17.8ms | — |
| convt3 | 25.6ms | 29.9ms | 20.5ms | 18.6ms | — | — | — |
| lstm | FAIL | FAIL | — | — | — | — | — |
| embedding | 377.9ms | 12.5ms | 10.0ms | 18.4ms | 11.5ms | 12.4ms | — |
| residual | 5429.2ms | 272871.2ms | 20.4ms | 8.4ms | 10.3ms | 16.8ms | — |
| swiglu | 155.7ms | 52.4ms | 58.1ms | 49.5ms | 59.9ms | 58.4ms | — |
| mha | 80.6ms | 61.5ms | 72.2ms | 66.2ms | 61.8ms | 68.0ms | — |

## Go→NPU overhead = call − DLL exec

cgo + pack/unpack (+ graph build on online layers).

| Layer | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |
|-------|-----|-----|----|----|-----|-----|------|
| dense | 168µs | 0µs | 395µs | 0µs | 339µs | 272µs | — |
| softmax | 29.4ms | 10.1ms | 16.0ms | 18.0ms | 24.7ms | 7.5ms | — |
| rmsnorm | 39.1ms | 31.8ms | 20.6ms | 12.9ms | 40.2ms | 29.4ms | — |
| layernorm | 21.1ms | 13.1ms | 20.2ms | 10.0ms | 12.9ms | 20.1ms | — |
| cnn2 | 31.3ms | 22.0ms | 37.0ms | 23.0ms | 20.7ms | 30.2ms | — |
| cnn3 | 42.3ms | 28.3ms | 45.0ms | 35.1ms | — | — | — |
| convt2 | 18.0ms | 28.4ms | 16.2ms | 16.1ms | 24.5ms | 17.3ms | — |
| convt3 | 25.1ms | 29.4ms | 20.0ms | 18.3ms | — | — | — |
| lstm | — | — | — | — | — | — | — |
| embedding | 376.0ms | 11.9ms | 9.6ms | 18.0ms | 10.9ms | 11.8ms | — |
| residual | 5427.6ms | 272865.0ms | 20.0ms | 7.9ms | 9.8ms | 16.4ms | — |
| swiglu | 155.1ms | 52.0ms | 57.5ms | 49.1ms | 59.5ms | 58.1ms | — |
| mha | 80.2ms | 61.2ms | 71.8ms | 65.8ms | 61.2ms | 67.5ms | — |

## Go vs native vs NPU — every layer × dtype

NPU column = DLL exec. **%** = `(slower−faster)/slower×100` vs that host.

| Layer | dtype | Go | Native | NPU | call | vs Go | vs Native | Fastest |
|-------|-------|-----|--------|-----|------|-------|-----------|--------|
| dense | f32 | 3.05ms | 9.64ms | 201µs | 369µs | NPU **93%** | NPU **98%** | **NPU** |
| dense | f16 | 3.05ms | 9.64ms | 194µs | 194µs | NPU **94%** | NPU **98%** | **NPU** |
| dense | i8 | 3.05ms | 9.64ms | 183µs | 578µs | NPU **94%** | NPU **98%** | **NPU** |
| dense | u8 | 3.05ms | 9.64ms | 164µs | 164µs | NPU **95%** | NPU **98%** | **NPU** |
| dense | i16 | 3.05ms | 9.64ms | 164µs | 503µs | NPU **95%** | NPU **98%** | **NPU** |
| dense | u16 | 3.05ms | 9.64ms | 249µs | 521µs | NPU **92%** | NPU **97%** | **NPU** |
| softmax | f32 | 4.31ms | 2.72ms | 656µs | 30.0ms | NPU **85%** | NPU **76%** | **NPU** |
| softmax | f16 | 4.31ms | 2.72ms | 446µs | 10.5ms | NPU **90%** | NPU **84%** | **NPU** |
| softmax | i8 | 4.31ms | 2.72ms | 485µs | 16.5ms | NPU **89%** | NPU **82%** | **NPU** |
| softmax | u8 | 4.31ms | 2.72ms | 453µs | 18.4ms | NPU **89%** | NPU **83%** | **NPU** |
| softmax | i16 | 4.31ms | 2.72ms | 785µs | 25.5ms | NPU **82%** | NPU **71%** | **NPU** |
| softmax | u16 | 4.31ms | 2.72ms | 530µs | 8.0ms | NPU **88%** | NPU **80%** | **NPU** |
| rmsnorm | f32 | 9.40ms | 2.43ms | 2.1ms | 41.2ms | NPU **78%** | NPU **15%** | **NPU** |
| rmsnorm | f16 | 9.40ms | 2.43ms | 1.7ms | 33.5ms | NPU **82%** | NPU **29%** | **NPU** |
| rmsnorm | i8 | 9.40ms | 2.43ms | 1.0ms | 21.6ms | NPU **89%** | NPU **57%** | **NPU** |
| rmsnorm | u8 | 9.40ms | 2.43ms | 897µs | 13.8ms | NPU **90%** | NPU **63%** | **NPU** |
| rmsnorm | i16 | 9.40ms | 2.43ms | 1.8ms | 42.0ms | NPU **81%** | NPU **27%** | **NPU** |
| rmsnorm | u16 | 9.40ms | 2.43ms | 1.3ms | 30.7ms | NPU **86%** | NPU **46%** | **NPU** |
| layernorm | f32 | 6.82ms | 2.60ms | 1.1ms | 22.2ms | NPU **84%** | NPU **58%** | **NPU** |
| layernorm | f16 | 6.82ms | 2.60ms | 553µs | 13.7ms | NPU **92%** | NPU **79%** | **NPU** |
| layernorm | i8 | 6.82ms | 2.60ms | 594µs | 20.8ms | NPU **91%** | NPU **77%** | **NPU** |
| layernorm | u8 | 6.82ms | 2.60ms | 635µs | 10.6ms | NPU **91%** | NPU **76%** | **NPU** |
| layernorm | i16 | 6.82ms | 2.60ms | 650µs | 13.6ms | NPU **90%** | NPU **75%** | **NPU** |
| layernorm | u16 | 6.82ms | 2.60ms | 587µs | 20.7ms | NPU **91%** | NPU **77%** | **NPU** |
| cnn2 | f32 | 12.56ms | — | 1.1ms | 32.4ms | NPU **91%** | — | **NPU** |
| cnn2 | f16 | 12.56ms | — | 833µs | 22.8ms | NPU **93%** | — | **NPU** |
| cnn2 | i8 | 12.56ms | — | 763µs | 37.8ms | NPU **94%** | — | **NPU** |
| cnn2 | u8 | 12.56ms | — | 1.0ms | 24.0ms | NPU **92%** | — | **NPU** |
| cnn2 | i16 | 12.56ms | — | 1.8ms | 22.5ms | NPU **86%** | — | **NPU** |
| cnn2 | u16 | 12.56ms | — | 1.1ms | 31.3ms | NPU **91%** | — | **NPU** |
| cnn3 | f32 | 1.33ms | — | 573µs | 42.9ms | NPU **57%** | — | **NPU** |
| cnn3 | f16 | 1.33ms | — | 509µs | 28.8ms | NPU **62%** | — | **NPU** |
| cnn3 | i8 | 1.33ms | — | 488µs | 45.5ms | NPU **63%** | — | **NPU** |
| cnn3 | u8 | 1.33ms | — | 647µs | 35.7ms | NPU **51%** | — | **NPU** |
| convt2 | f32 | 3.14ms | — | 724µs | 18.7ms | NPU **77%** | — | **NPU** |
| convt2 | f16 | 3.14ms | — | 542µs | 28.9ms | NPU **83%** | — | **NPU** |
| convt2 | i8 | 3.14ms | — | 603µs | 16.8ms | NPU **81%** | — | **NPU** |
| convt2 | u8 | 3.14ms | — | 620µs | 16.8ms | NPU **80%** | — | **NPU** |
| convt2 | i16 | 3.14ms | — | 877µs | 25.4ms | NPU **72%** | — | **NPU** |
| convt2 | u16 | 3.14ms | — | 506µs | 17.8ms | NPU **84%** | — | **NPU** |
| convt3 | f32 | 285µs | — | 499µs | 25.6ms | host **43%** | — | **Go** |
| convt3 | f16 | 285µs | — | 510µs | 29.9ms | host **44%** | — | **Go** |
| convt3 | i8 | 285µs | — | 424µs | 20.5ms | host **33%** | — | **Go** |
| convt3 | u8 | 285µs | — | 378µs | 18.6ms | host **25%** | — | **Go** |
| lstm | f32 | 1.38ms | — | FAIL | FAIL | — | — | **Go** |
| lstm | f16 | 1.38ms | — | FAIL | FAIL | — | — | **Go** |
| embedding | f32 | 69µs | 82µs | 1.9ms | 377.9ms | host **96%** | host **96%** | **Go** |
| embedding | f16 | 69µs | 82µs | 572µs | 12.5ms | host **88%** | host **86%** | **Go** |
| embedding | i8 | 69µs | 82µs | 435µs | 10.0ms | host **84%** | host **81%** | **Go** |
| embedding | u8 | 69µs | 82µs | 464µs | 18.4ms | host **85%** | host **82%** | **Go** |
| embedding | i16 | 69µs | 82µs | 566µs | 11.5ms | host **88%** | host **85%** | **Go** |
| embedding | u16 | 69µs | 82µs | 524µs | 12.4ms | host **87%** | host **84%** | **Go** |
| residual | f32 | 91µs | — | 1.6ms | 5429.2ms | host **94%** | — | **Go** |
| residual | f16 | 91µs | — | 6.2ms | 272871.2ms | host **99%** | — | **Go** |
| residual | i8 | 91µs | — | 470µs | 20.4ms | host **81%** | — | **Go** |
| residual | u8 | 91µs | — | 417µs | 8.4ms | host **78%** | — | **Go** |
| residual | i16 | 91µs | — | 438µs | 10.3ms | host **79%** | — | **Go** |
| residual | u16 | 91µs | — | 434µs | 16.8ms | host **79%** | — | **Go** |
| swiglu | f32 | 1.92ms | 5.49ms | 589µs | 155.7ms | NPU **69%** | NPU **89%** | **NPU** |
| swiglu | f16 | 1.92ms | 5.49ms | 378µs | 52.4ms | NPU **80%** | NPU **93%** | **NPU** |
| swiglu | i8 | 1.92ms | 5.49ms | 666µs | 58.1ms | NPU **65%** | NPU **88%** | **NPU** |
| swiglu | u8 | 1.92ms | 5.49ms | 362µs | 49.5ms | NPU **81%** | NPU **93%** | **NPU** |
| swiglu | i16 | 1.92ms | 5.49ms | 386µs | 59.9ms | NPU **80%** | NPU **93%** | **NPU** |
| swiglu | u16 | 1.92ms | 5.49ms | 352µs | 58.4ms | NPU **82%** | NPU **94%** | **NPU** |
| mha | f32 | 4.51ms | 5.49ms | 439µs | 80.6ms | NPU **90%** | NPU **92%** | **NPU** |
| mha | f16 | 4.51ms | 5.49ms | 372µs | 61.5ms | NPU **92%** | NPU **93%** | **NPU** |
| mha | i8 | 4.51ms | 5.49ms | 382µs | 72.2ms | NPU **92%** | NPU **93%** | **NPU** |
| mha | u8 | 4.51ms | 5.49ms | 386µs | 66.2ms | NPU **91%** | NPU **93%** | **NPU** |
| mha | i16 | 4.51ms | 5.49ms | 530µs | 61.8ms | NPU **88%** | NPU **90%** | **NPU** |
| mha | u16 | 4.51ms | 5.49ms | 413µs | 68.0ms | NPU **91%** | NPU **92%** | **NPU** |

## Stack total — all layers on Go vs native vs NPU

Sum of per-layer exec (skip missing). Go/native summed once per layer (f32). NPU summed per dtype.

| Backend | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |
|---------|-----|-----|----|----|-----|-----|------|
| Go f32 (13 layers) | 48.88ms (13) | 48.88ms (13) | 48.88ms (13) | 48.88ms (13) | 48.88ms (13) | 48.88ms (13) | 48.88ms (13) |
| native f32 (7 layers) | 28.44ms (7) | 28.44ms (7) | 28.44ms (7) | 28.44ms (7) | 28.44ms (7) | 28.44ms (7) | 28.44ms (7) |
| NPU DLL | 11.46ms (12) | 12.82ms (12) | 6.53ms (12) | 6.43ms (12) | 7.93ms (10) | 6.04ms (10) | — |
| NPU call | 6256.82ms (12) | 273165.98ms (12) | 340.91ms (12) | 280.67ms (12) | 272.86ms (10) | 264.62ms (10) | — |

## NPU wins only (subset)

| Layer | dtype | Go | Native | NPU | vs Go | vs Native |
|-------|-------|-----|--------|-----|-------|----------|
| dense | f32 | 3.05ms | 9.64ms | 201µs | NPU **93%** | NPU **98%** |
| dense | f16 | 3.05ms | 9.64ms | 194µs | NPU **94%** | NPU **98%** |
| dense | i8 | 3.05ms | 9.64ms | 183µs | NPU **94%** | NPU **98%** |
| dense | u8 | 3.05ms | 9.64ms | 164µs | NPU **95%** | NPU **98%** |
| dense | i16 | 3.05ms | 9.64ms | 164µs | NPU **95%** | NPU **98%** |
| dense | u16 | 3.05ms | 9.64ms | 249µs | NPU **92%** | NPU **97%** |
| softmax | f32 | 4.31ms | 2.72ms | 656µs | NPU **85%** | NPU **76%** |
| softmax | f16 | 4.31ms | 2.72ms | 446µs | NPU **90%** | NPU **84%** |
| softmax | i8 | 4.31ms | 2.72ms | 485µs | NPU **89%** | NPU **82%** |
| softmax | u8 | 4.31ms | 2.72ms | 453µs | NPU **89%** | NPU **83%** |
| softmax | i16 | 4.31ms | 2.72ms | 785µs | NPU **82%** | NPU **71%** |
| softmax | u16 | 4.31ms | 2.72ms | 530µs | NPU **88%** | NPU **80%** |
| rmsnorm | f32 | 9.40ms | 2.43ms | 2.1ms | NPU **78%** | NPU **15%** |
| rmsnorm | f16 | 9.40ms | 2.43ms | 1.7ms | NPU **82%** | NPU **29%** |
| rmsnorm | i8 | 9.40ms | 2.43ms | 1.0ms | NPU **89%** | NPU **57%** |
| rmsnorm | u8 | 9.40ms | 2.43ms | 897µs | NPU **90%** | NPU **63%** |
| rmsnorm | i16 | 9.40ms | 2.43ms | 1.8ms | NPU **81%** | NPU **27%** |
| rmsnorm | u16 | 9.40ms | 2.43ms | 1.3ms | NPU **86%** | NPU **46%** |
| layernorm | f32 | 6.82ms | 2.60ms | 1.1ms | NPU **84%** | NPU **58%** |
| layernorm | f16 | 6.82ms | 2.60ms | 553µs | NPU **92%** | NPU **79%** |
| layernorm | i8 | 6.82ms | 2.60ms | 594µs | NPU **91%** | NPU **77%** |
| layernorm | u8 | 6.82ms | 2.60ms | 635µs | NPU **91%** | NPU **76%** |
| layernorm | i16 | 6.82ms | 2.60ms | 650µs | NPU **90%** | NPU **75%** |
| layernorm | u16 | 6.82ms | 2.60ms | 587µs | NPU **91%** | NPU **77%** |
| cnn2 | f32 | 12.56ms | — | 1.1ms | NPU **91%** | — |
| cnn2 | f16 | 12.56ms | — | 833µs | NPU **93%** | — |
| cnn2 | i8 | 12.56ms | — | 763µs | NPU **94%** | — |
| cnn2 | u8 | 12.56ms | — | 1.0ms | NPU **92%** | — |
| cnn2 | i16 | 12.56ms | — | 1.8ms | NPU **86%** | — |
| cnn2 | u16 | 12.56ms | — | 1.1ms | NPU **91%** | — |
| cnn3 | f32 | 1.33ms | — | 573µs | NPU **57%** | — |
| cnn3 | f16 | 1.33ms | — | 509µs | NPU **62%** | — |
| cnn3 | i8 | 1.33ms | — | 488µs | NPU **63%** | — |
| cnn3 | u8 | 1.33ms | — | 647µs | NPU **51%** | — |
| convt2 | f32 | 3.14ms | — | 724µs | NPU **77%** | — |
| convt2 | f16 | 3.14ms | — | 542µs | NPU **83%** | — |
| convt2 | i8 | 3.14ms | — | 603µs | NPU **81%** | — |
| convt2 | u8 | 3.14ms | — | 620µs | NPU **80%** | — |
| convt2 | i16 | 3.14ms | — | 877µs | NPU **72%** | — |
| convt2 | u16 | 3.14ms | — | 506µs | NPU **84%** | — |
| swiglu | f32 | 1.92ms | 5.49ms | 589µs | NPU **69%** | NPU **89%** |
| swiglu | f16 | 1.92ms | 5.49ms | 378µs | NPU **80%** | NPU **93%** |
| swiglu | i8 | 1.92ms | 5.49ms | 666µs | NPU **65%** | NPU **88%** |
| swiglu | u8 | 1.92ms | 5.49ms | 362µs | NPU **81%** | NPU **93%** |
| swiglu | i16 | 1.92ms | 5.49ms | 386µs | NPU **80%** | NPU **93%** |
| swiglu | u16 | 1.92ms | 5.49ms | 352µs | NPU **82%** | NPU **94%** |
| mha | f32 | 4.51ms | 5.49ms | 439µs | NPU **90%** | NPU **92%** |
| mha | f16 | 4.51ms | 5.49ms | 372µs | NPU **92%** | NPU **93%** |
| mha | i8 | 4.51ms | 5.49ms | 382µs | NPU **92%** | NPU **93%** |
| mha | u8 | 4.51ms | 5.49ms | 386µs | NPU **91%** | NPU **93%** |
| mha | i16 | 4.51ms | 5.49ms | 530µs | NPU **88%** | NPU **90%** |
| mha | u16 | 4.51ms | 5.49ms | 413µs | NPU **91%** | NPU **92%** |

## Failures

- **lstm/float32**: `nativeqnn lstm: code=2 execute`
- **lstm/float16**: `nativeqnn lstm: code=2 execute`

## Re-run

```bash
cd w2a
go test ./tests/go_v_qnn -v -count=1 -run TestGoVQNNMatrix -timeout 180m
```
