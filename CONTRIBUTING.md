# Changing the compiler or objective

A tool change starts in `src/tool_calling_lab/robot.py`. Give it a typed input contract, a bounded transition, and a case whose independently constructed final state catches a bad implementation. Keep exact-call matching separate from equivalent movement partitioning.

For training changes, add a small numerical example before touching the optimizer. Tests should establish the causal label shift, prompt/EOS masking, frozen-reference behavior and resume identity. Do not replace an unavailable model with the reference compiler. The `train`, `dpo-train` and evaluation commands must continue to report which path actually ran.

From an isolated Python environment:

```bash
python -m pip install -e . -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Install `.[ml]` and set `RUN_ML_SMOKE=1` to include the tiny local-model checks. `python bin/check_training.py --require-lint --wheel` also checks the packaged command outside the checkout.

An experiment change should include its suite, split, seed, model identity and raw failed outputs. Keep training and validation choices away from final test compositions. Store large weights outside this Git repository; include their content identities with the run record instead.
