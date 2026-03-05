from __future__ import annotations

import argparse
import json
from pathlib import Path

from .robot import evaluate, generate_composition_dataset, generate_dataset, reference_compile


def _write(report: object, output: str | None) -> None:
    value = json.dumps(report, indent=2) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    else:
        print(value, end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled-language tool execution and optional real local LoRA experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "dataset", "hf-evaluate", "train"):
        p = sub.add_parser(name)
        p.add_argument("--output")
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--count", type=int, default=36 if name != "demo" else 3)
        p.add_argument(
            "--split",
            choices=("train",) if name == "train" else ("train", "validation", "test"),
            default="train" if name == "train" else "test",
        )
        p.add_argument("--suite", choices=("primitive", "compositional"), default="compositional")
        if name in {"train", "hf-evaluate"}:
            p.add_argument("--model", required=True)
        if name == "train":
            p.add_argument("--steps", type=int, default=20)
        if name == "hf-evaluate":
            p.add_argument("--adapter")
            p.add_argument("--max-new-tokens", type=int, default=256)
    p = sub.add_parser("preferences", help="Generate independently execution-ranked synthetic preference pairs")
    p.add_argument("--split", choices=("train", "validation", "test"), default="train")
    p.add_argument("--count", type=int, default=36)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output")
    p = sub.add_parser("dpo-train", help="Actually optimize a local LoRA policy with completion-only DPO")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial-adapter")
    p.add_argument("--resume", help="Committed checkpoint directory or run directory")
    p.add_argument("--count", type=int, default=36)
    p.add_argument("--steps", type=int, default=20, help="Desired total optimizer steps, including resumed steps")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p = sub.add_parser("dpo-evaluate", help="Score held-out preferences and execute actual generated model outputs")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "preferences":
        from .preferences import preference_report

        _write(preference_report(args.split, args.count, args.seed), args.output)
        return
    if args.command == "dpo-train":
        from .dpo import train_dpo

        report = train_dpo(
            args.model,
            args.output,
            args.count,
            args.steps,
            args.seed,
            args.beta,
            args.learning_rate,
            args.checkpoint_every,
            args.resume,
            args.initial_adapter,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "dpo-evaluate":
        from .dpo import evaluate_dpo

        _write(
            evaluate_dpo(args.model, args.checkpoint, args.split, args.count, args.seed, args.max_new_tokens),
            args.output,
        )
        return
    if args.command == "train":
        if not args.output:
            parser.error("train requires --output for adapter directory")
        from .hf import train

        report = train(args.model, args.output, args.count, args.steps, args.seed, args.suite)
        print(json.dumps(report, indent=2))
        return
    cases = (generate_composition_dataset if args.suite == "compositional" else generate_dataset)(
        args.split, args.count, args.seed
    )
    if args.command == "dataset":
        _write(
            {
                "source": "synthetic",
                "split": args.split,
                "suite": args.suite,
                "seed": args.seed,
                "cases": [case.to_dict() for case in cases],
            },
            args.output,
        )
        return
    if args.command == "hf-evaluate":
        from .hf import infer

        outputs = infer(args.model, args.adapter, cases, args.max_new_tokens)
        report = evaluate(
            cases, outputs, "actual local HF model" + (" with LoRA adapter" if args.adapter else " without adapter")
        )
        report["model_path"] = args.model
        report["adapter"] = args.adapter
        report["learned_model_evaluated"] = True
    else:
        outputs = [json.dumps(reference_compile(case.instruction)) for case in cases]
        report = evaluate(cases, outputs, "deterministic handwritten reference, NOT trained-model accuracy")
        report["unrun_workflows"] = ["base-model evaluation", "LoRA training", "adapter evaluation"]
        report["learned_model_evaluated"] = False
    report.update({"split": args.split, "seed": args.seed})
    _write(report, args.output)
