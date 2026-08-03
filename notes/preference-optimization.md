# Execution-derived DPO: algorithm and implementation

## Scope

This repository now contains two actual optimization paths: completion-only supervised LoRA training and direct preference optimization. DPO is implemented with explicit sequence likelihoods, a frozen reference, optimizer state and resumable checkpoints. Default stdlib commands still require no model, training dependency, service or GPU.

## 1. Independent preference supervision

The compositional generator defines task intent, initial world, golden calls and independently constructed final-state labels. Candidate outputs are parsed and executed, then compared with those labels and canonical action traces. Reward is:

`schema_valid + executable + 2 * final_state_success + 4 * instruction_success`.

The resulting tiers are 0 (malformed), 1 (valid schema but execution failure), 2 (executable wrong task), 4 (same final state but wrong required actions), and 8 (correct action semantics). The generator retains only strict chosen > rejected pairs, never ties. It includes wrong distances/directions/items and format failures. Positive movement partitioning is permitted only within the same split's allowed action signatures. Schema-valid negatives are similarly restricted; requests and successful targets do not leak held-out operation compositions into training.

The preference source is synthetic and solver-supervised. It does not claim that these candidates are sampled from a model or represent natural human preferences. The complete generated payload is hashed and recorded.

## 2. Completion-only causal likelihood

For prompt x and completion y, the scorer sums `log p_theta(y_t | x, y_<t)` over completion tokens including EOS. It does not average by sequence length. In causal tensors, labels at index j are gathered from logits at j-1. Prompt labels and right-padding labels are -100, and the attention mask independently excludes padding. This prevents prompt tokens or padding EOS IDs from leaking into the preference objective.

Chosen and rejected completions form a two-row padded batch. Both use exactly the same prompt. Context overflow is rejected; targets are never silently truncated. The implementation uses float32 log-softmax for numerical stability. Sequence-length bias is an explicit property of this standard summed-likelihood objective; it is not concealed with a post-hoc length bonus.

## 3. DPO objective and frozen reference

Let `delta = (log pi(chosen) - log pi(rejected)) - (log ref(chosen) - log ref(rejected))`.

The batch loss is `mean(softplus(-beta * delta))`, equivalent to negative log-sigmoid but stable for large positive/negative margins. Beta must be positive and finite. Reference tensors are detached; reference parameters additionally have `requires_grad=False` and never enter the optimizer.

The policy is a trainable LoRA adapter over a local causal model. Without an initial adapter, the reference is the base checkpoint. With an initial SFT adapter, both policy and reference begin from that snapshot, and only policy adapter parameters update. A separate reference model avoids adapter-switching or shared-autograd ambiguity. This costs approximately two base-model copies in memory.

Both models use eval mode during preference optimization to disable dropout; eval mode does not disable policy autograd. AdamW parameters, clipping, beta, dataset identity, model/tokenizer content and dependency versions are explicit configuration. No reward model, PPO loop or TRL wrapper is involved.

## 4. Resumable state machine

```mermaid
flowchart LR
  P[Strict execution-ranked pairs] --> F[Policy and frozen-reference forward passes]
  F --> O[Completion log ratios and stable DPO loss]
  O --> U[LoRA gradient update]
  U --> S[Write pending generation]
  S --> V[Adapter, tokenizer, optimizer and RNG hashes]
  V --> C[Atomic directory rename commits step]
  C --> R[Resume verifies data, model and config]
  R --> F
```

A generation contains adapter/tokenizer files, `state.pt` (optimizer, CPU/CUDA/Python RNG, completed step, sample history), and a file-hashed manifest. Files and directories are synchronized before the directory is atomically renamed to `step-NNNNNN`. `latest.json` is an advisory pointer. Recovery scans committed generations, ignoring pending directories and stale pointers. A corrupt newest generation fails integrity validation instead of silently rolling back an experiment.

Resume binds data hash, base/tokenizer content hash, initial reference-adapter hash, context, beta, AdamW settings, seed, device and ML-library versions. Total requested steps and checkpoint frequency do not affect optimizer mathematics and may change. Data order is a deterministic epoch shuffle derived from seed and the saved global step. Optimizer/RNG state is restored after model construction. Tensor state loads with `weights_only=True`.

This is a single-writer local-filesystem protocol. It is not a distributed lock, remote object-store transaction or power-loss certification. The tested CPU resume matches uninterrupted adapter weights bitwise. GPU determinism is unverified.

## 5. Evaluation

`dpo-evaluate` loads a committed checkpoint and its unchanged reference, then evaluates validation or test compositions. It reports positive reference-adjusted preference margins, ties, and actual generated-output execution metrics. Generation uses a separate explicit token budget and strict JSON/simulator validation. No training update or threshold selection occurs during evaluation. Training does not automatically evaluate test data.

## 6. Complexity and tradeoffs

Each optimizer step performs one chosen/rejected policy batch forward/backward and one frozen-reference forward. Both full sequence vocabulary-logit tensors are materialized; runtime scales with sequence/model size and memory includes two models plus the policy autograd graph. LoRA reduces trainable/optimizer state but does not eliminate reference/base activation memory. Reference likelihoods could be precomputed for a fixed dataset; this implementation deliberately recomputes them to keep provenance direct.

There is no distributed training, mixed precision, gradient accumulation, reference log-probability cache or learning-rate scheduler. The implementation targets understandable, independently tested mathematics and durable local experiments. Large-model throughput and useful adaptation require separate measurements.
