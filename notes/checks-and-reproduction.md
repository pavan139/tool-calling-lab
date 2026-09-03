# Checks for robot programs and optimizer state

The recorded full run executed 40 tests on Python 3.12.14, including five opt-in ML checks. The default stdlib run explicitly skips those five checks. The detailed result and source identity are in [the verification artifact](../reports/verification.json).

```bash
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s tests -v
python bin/check_training.py --require-lint --wheel
```

The tests cover JSON/type boundaries, independently labeled goals, action-composition partitions, causal completion likelihoods, DPO gradients, frozen references and optimizer/RNG restoration. The tiny CPU experiment uses random weights and a deliberately short generation budget. [The training record](dpo-training-record.md) preserves its zero held-out execution result rather than treating a successful optimizer step as an accuracy result.

The verifier additionally builds a wheel and imports it from outside the source tree. Check the artifact's source hash when comparing results across edits. Pretrained-model accuracy, broad tokenizer support and GPU measurements require separate runs.
