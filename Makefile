.PHONY: test demo evaluate
test:
	PYTHONPATH=src python -m unittest discover -s tests -v
demo:
	PYTHONPATH=src python -m tool_calling_lab demo --output artifacts/demo.json
evaluate:
	PYTHONPATH=src python -m tool_calling_lab evaluate --count 36 --output artifacts/reference.json

.PHONY: preferences dpo-smoke
preferences:
	PYTHONPATH=src python -m tool_calling_lab preferences --count 36 --output artifacts/preferences.json
dpo-smoke:
	PYTHONPATH=src python bin/dpo_smoke.py --output artifacts/dpo-smoke.json
