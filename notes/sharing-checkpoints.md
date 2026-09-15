# Sharing a tool-calling experiment

The small files in `recipes/` and `samples/` are enough to reproduce the reference experiment. Model weights are a separate dependency. A useful training release includes the base-checkpoint identity, optional SFT reference-adapter identity, generated preference hash, optimizer settings and the evaluation JSON. A path on one machine is not a portable model identity.

Run `python bin/check_training.py --require-lint --wheel` before packaging the source. To include local-model integration, install the `ml` extra and run the test suite with `RUN_ML_SMOKE=1`. The random-checkpoint smoke checks the training code; it does not establish useful model quality.

For an extracted folder without Git metadata, initialize it with `git init -b main`, inspect `git status`, and choose which source, notes and small run records to track. Do not include `artifacts/`, environments or model checkpoints. If this folder already has a Git repository, keep that history and configuration.

After making your own commit, connect an empty GitHub repository with its actual URL:

```bash
git remote add origin https://github.com/YOUR_USERNAME/tool-calling-lab.git
git push -u origin main
```

GitHub Actions runs the lightweight package checks after a push. Keep the executed model experiment and the reference-compiler result labeled separately in the release description.
