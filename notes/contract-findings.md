# Review: contracts, generalization, scoring and local-model integration

This review inspected the generator, simulator, JSON boundary, scorer, CLI, training labels, context accounting, tokenizer compatibility and optional HF/PEFT path. All results below are controlled synthetic or random-checkpoint evidence, not pretrained-model accuracy.

## Reproduced defects and fixes

| Reproduction | Before | Correction and regression evidence |
| --- | --- | --- |
| A JSON response with a 5,000-digit integer, below the output-byte limit | Python's integer-conversion `ValueError` escaped the evaluator | Parsing catches the decoder's value/recursion errors and reports a `ContractError`; the evaluator records a failed case instead of aborting the batch. |
| A model string containing a lone Unicode surrogate | `UnicodeEncodeError` escaped before JSON validation | Invalid UTF-8 text fails explicitly at the output boundary. |
| `World(obstacles=None)` or a non-State initial object | Raw `TypeError`/attribute errors | Typed records, obstacle containers and item/resource bounds are validated. |
| Train/test action signatures in the original generator | All three signatures (`M`, `MP`, `MPMD`) appeared in both splits | A separate compositional suite, now the CLI default, holds out action signatures as well as phrases and items. Primitive coverage remains available with `--suite primitive`. |
| Golden and predicted programs scored only through the same executor | A shared transition bug could contaminate both sides | Task construction independently labels final states; golden execution must match those labels before prediction scoring. |
| Skip a required PICK/DROP pair while returning to the same state | Final-state success alone credited the alias | New `instruction_success` also compares canonical action traces, accepting split moves but rejecting skipped effects. Final-state success remains explicitly separate. |
| Checkpoint generation settings or special output tokens | Inherited settings could alter decoding; `skip_special_tokens=True` removed observable generated tokens | Fresh greedy settings disable inherited model defaults on supported Transformers versions; only terminal EOS is removed. A real tiny-checkpoint test verifies generated BOS text is retained. |

Context accounting now uses the tightest real model/tokenizer limit, ignores sentinel tokenizer limits, validates vocabulary/EOS against embeddings, and tests completion-only labels without target/EOS truncation. The training CLI only permits the train split. Evaluations reject duplicate case IDs and hash complete case labels.

## High-value capability

The new compositional suite has these disjoint action families:

- Train: `M`, `MP`, `MPMD`.
- Validation: `MMP`, `MPM`.
- Test: `MMPMD`, `MPMMD`, `MMPMMD`.

This is a controlled compositional transfer experiment. It does not claim unrestricted natural-language generalization. Every generated case's independently constructed goal is verified in tests, across 72 cases per split. The handwritten reference deliberately knows every grammar family.

## Commands actually run

From this project directory, using an environment with the optional ML dependencies already installed:

```bash
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m tool_calling_lab evaluate --count 36 --output samples/reference-evaluation.json
PYTHONPATH=src python -m tool_calling_lab demo --output samples/demo.json
```

The complete suite passed **26/26 tests**, including the opt-in local integration smoke. Without `RUN_ML_SMOKE=1`, the same discovery runs 25 core tests and explicitly skips the integration test. Ruff checks and formatting checks passed on the project code.

The 36-case compositional **handwritten reference** run recorded 36/36 schema-valid, executable, exact-call, final-state-success and instruction-success outputs. Its dataset SHA-256 is `028196016628fea489534c14e21877326041276bcd14a0ab47e470c06d457e40`. These numbers validate the reference and dataset pipeline, not learned-model accuracy.

The integration test creates a random tiny GPT-2 and a locally trained byte-level tokenizer, performs exactly **one LoRA optimizer step on two examples**, saves/reloads the adapter and executes bounded inference. It also verifies raw special-token preservation under an inherited forced-EOS/minimum-length configuration. It used CPU PyTorch 2.6.0 and Transformers 4.57.1, with no downloaded weights or paid calls. `samples/integration-smoke.json` records the separate executed smoke; it is not a useful training run.

## Remaining limits

- Only a narrow controlled language and fictional tools are evaluated; obstacle planning and open-ended dialogue are excluded.
- Broad model/tokenizer/adapter compatibility is not established by a tiny GPT-2 smoke.
- No meaningful pretrained training run, pretrained-model accuracy, GPU memory study or inference-performance comparison has been performed.
- Dataset labels and the simulator are independent implementations within this repository, not externally audited ground truth.
- Local checkpoint identity and execution configuration must be archived when publishing meaningful model results; path names alone do not attest model provenance.
