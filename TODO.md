# Road to a small LLM in Scanet (alongside Stanford CS336)

Goal: understand how LLMs work by building a small decoder-only transformer language model here.
Pace: one CS336 lecture at a time. After each lecture, write the code yourself, ask for generated parts only
where it is not the point of the lesson. One small PR per step. Update the checkboxes when a step is merged.

The lecture order below follows CS336 (Spring 2025) roughly. Check it against the playlist you watch and
move steps around if needed. CS336 assignments use PyTorch. Systems lectures (GPU kernels, FlashAttention,
FSDP) will not map to TF Java + Spark, so for those the goal is the concept, not the code.

## 0. Preparation (done)

- [x] TF Java 1.2.0 (official macOS arm64), Spark 3.5, sbt 1.10, Scala 2.12 + 2.13 cross-build (#100)
- [x] Batched `MatMul` for rank >= 2 with broadcast batch dims (#100)
- [x] Fix `Transpose` gradient for custom permutations, fix `BatchNorm` penalty (#100)
- [x] Nested array/sequence tensor literals: `Tensor(Array(Array(1, 2), Array(3, 4)))` (#101)

## 1. Fix known bugs (before new work)

Fix each one with a test that compares the result with the reference formula,
not only that the loss goes down.

- [ ] `ReLU(threshold)` returns `threshold` below the threshold, the usual definition returns 0
      (needs `where`, comparison ops have no gradient, so fix it together with `where` in step 5)
- [ ] `Softmax` is not numerically stable and works on axis 1 only (covered again in step 5)
- [ ] `BatchNorm` uses `trainable` as the training mode flag, so "frozen" and "inference" mean the same.
      Dropout needs a separate training / inference mode

## 2. Refresh (before lecture 1)

Read `docs/` and the code only where the recap points to it.

- [ ] Linear algebra: shapes, broadcasting, matmul, transpose, reductions (`math/alg/AllKernels.scala`)
- [ ] Backprop: how `grad` walks the graph and what `localGrad` is (`math/grad/package.scala`, `core/Expr.scala`)
- [ ] Models built so far: linear / logistic regression, MLP, CNN, RNN/LSTM/GRU, BatchNorm
- [ ] Training: losses, initializers, SGD to Adam, regularization, Spark model averaging (`optimizers/`)
- [ ] Re-run `ANNSpec` and the MNIST benchmark to see the training loop work end to end

## 3. Lecture 1: overview and tokenization

- [ ] Byte-level BPE tokenizer: train (count pairs, merge), encode, decode, special tokens
- [ ] Round trip tests (`decode(encode(text)) == text`), compression ratio on a small corpus
- [ ] Token ids to `Dataset[Record[Float]]` (fixed length context windows, next token as the label)

## 4. Lecture 2: resource accounting

- [ ] Count params, FLOPs and memory for `Dense`, attention and a full block by hand
- [ ] Add a parameter count and rough FLOPs line to `Model.describe`
- [ ] Float32 vs float16/bfloat16: what TF Java supports, what is worth trying

## 5. Lecture 3: architecture and hyperparameters

Ops (each new primitive needs a gradient and a test, compositions get gradients for free):

- [ ] `gather` (embedding lookup) with gradient
- [ ] `where` / masking with gradient
- [ ] Stable `softmax(axis)` (subtract max), `logSoftmax`
- [ ] Cross-entropy from logits over a vocabulary (`SparseCategoricalCrossentropy`)
- [ ] `LayerNorm` and `RMSNorm` (compose from `mean`, `sqrt`)
- [ ] Activations: GELU, SiLU / SwiGLU
- [ ] `Dropout` with a random generator (the old README "wanted" item)

Layers:

- [ ] `Embedding`
- [ ] `Residual(layer)` combinator
- [ ] Rotary position embeddings (RoPE), sinusoidal as a warm-up
- [ ] Scaled dot-product attention with a causal mask
- [ ] `MultiHeadAttention` (split heads with `reshape` + `transpose`)
- [ ] Position-wise feed forward (SwiGLU)
- [ ] `TransformerBlock` (pre-norm), then `TransformerLM`

## 6. Training the LM (assignment 1 equivalent)

- [ ] AdamW (decoupled weight decay)
- [ ] Gradient clipping by global norm
- [ ] Learning rate schedule: linear warmup + cosine decay
- [ ] Gradient averaging every step instead of weight averaging every epoch (Spark)
- [ ] Model export / import (checkpoints)
- [ ] Train a tiny char-level LM, then a small BPE-token LM, track loss and perplexity in TensorBoard
- [ ] Sampling: greedy, temperature, top-k / top-p

## 7. Later lectures (concepts first, code only if it fits)

- [ ] GPUs, kernels, FlashAttention: read, maybe benchmark attention on CPU vs GPU
- [ ] Parallelism (data, tensor, pipeline): compare with what the Spark optimizer does
- [ ] Scaling laws: run a small sweep of model sizes, plot loss vs compute
- [ ] Inference: KV cache
- [ ] Mixture of experts: a small MoE feed forward layer
- [ ] Data: filtering and deduplication on Spark
- [ ] Alignment: SFT, then the basic idea of RLHF / GRPO

## Known gaps in the existing code

- Exact float comparisons in tests are brittle across TF versions (use `beCloseTo`)
- Initializers are not seeded in tests, so convergence tests can be flaky
  (`ANNSpec` "minimize logistic regression" failed once with loss 0.64 > 0.5, then passed 3 of 3)
- No model export / import, no dropout, only the MNIST dataset
- `Optimizer` averages weights once per epoch, this hurts convergence for deep models
