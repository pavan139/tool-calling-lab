# DPO optimizer and resume experiment

The preference path computes its own likelihood objective and saves resumable optimizer state alongside the SFT workflow. The results below describe the local experiment that was run.

## What the run exercises

- **Execution-derived data:** generate strict chosen/rejected pairs from independent final-state and action-trace labels. Candidate faults include valid wrong arguments and malformed outputs. Ties are discarded. Successful targets and schema-valid negative compositions remain within their assigned split; no test instruction family is trained on.
- **Explicit DPO mathematics:** compute causal, completion-only sequence log probabilities with the correct one-token shift, prompt masking, attention/padding masking and target EOS. Optimize a stable softplus reference-adjusted preference loss.
- **Actual trainable policy:** update LoRA parameters with AdamW against a separately loaded, frozen reference. An optional SFT adapter can initialize both the policy and frozen reference. Dropout is disabled without disabling policy gradients.
- **Durable state:** atomically commit adapter, tokenizer, optimizer, Python/Torch RNG, step and sample history. Data, model/tokenizer/reference-adapter contents, hyperparameters, context, device and library versions bind resume. Pending generations are ignored and corrupt files reject loading.
- **Actual held-out evaluation:** reload saved adapters, score reference-adjusted preference margins and separately execute generated tool calls. Validation and test compositions remain distinct from training.

See [preference-optimization.md](preference-optimization.md) for equations, checkpoint transitions, complexity and tradeoffs.

## Commands actually executed

With optional dependencies already present in the selected CPU environment:

```bash
PYTHONPATH=src python -m tool_calling_lab preferences --split train --count 12 --output samples/preference-sample.json
python bin/dpo_smoke.py --output samples/dpo-integration-smoke.json
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s tests -v
python bin/check_training.py --require-lint --wheel --output reports/depth-verification.json
```

The complete suite passed **40/40 tests**, including five explicitly enabled ML integration tests. Without `RUN_ML_SMOKE=1`, 35 stdlib tests pass and five optional tests are reported as skipped. The DPO-specific integration tests cover the causal shift/padding gradient, tensor loss against an independent scalar oracle, detached reference gradients, actual optimization, resume equivalence, changed-beta/data rejection, saved-model evaluation and SFT-reference initialization.

## Measured random-model experiment

The committed `samples/dpo-integration-smoke.json` is an executed CPU experiment using a locally created random one-layer GPT-2 (embedding width 16, two attention heads, vocabulary 457). No pretrained checkpoint or paid service was used.

| Observation | Executed result |
| --- | --- |
| Training preference corpus | 4 synthetic execution-ranked pairs |
| Actual optimizer steps | 2 |
| Step losses | 0.6931471825, 0.6931441426 |
| Nonzero gradient norms | 0.0414367244, 0.1422250867 |
| Interrupted/resumed versus uninterrupted adapter | Bitwise equal |
| Interrupted/resumed sample/loss history | Equal |
| Reference parameters with gradients | 0 |
| Held-out validation preferences | 0/2 preferred |
| Generated execution success, four-token budget | 0/2 |

Losses come from different sampled pairs and do not establish a learning trend. The zero held-out result is intentionally retained. This experiment proves that optimization, persistence and evaluation actually execute; it establishes no useful pretrained capability, adaptation gain or generalization.

## Explicit limits

- Synthetic solver-ranked pairs are not human preference data or representative model-generated failures.
- Two base-model copies are kept in memory. There is no distributed training, mixed precision, gradient accumulation or reference-likelihood cache.
- Summed completion likelihood has sequence-length effects; no hidden normalization or benchmark-tuned reward disguises them.
- Atomic checkpoint behavior targets one writer on a local filesystem. There is no distributed lease or power-loss certification. CPU bitwise resume was tested; GPU determinism was not.
- The base checkpoint and any initial SFT reference adapter remain external content-hashed dependencies. Moving files is permissible if their content is unchanged; removing the reference dependency prevents evaluation.
- No meaningful pretrained DPO run, GPU throughput experiment or model-quality improvement has been measured.
