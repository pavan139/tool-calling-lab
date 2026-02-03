"""Optional real local-checkpoint workflows; imports never affect the stdlib core."""

from __future__ import annotations

import json
import hashlib
import inspect
from pathlib import Path

from .robot import Case, format_prompt, generate_composition_dataset, generate_dataset


def context_limit(config: object, fallback: int = 1024, tokenizer: object | None = None) -> int:
    limits = []
    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, key, None)
        if type(value) is int and value > 0:
            limits.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if type(tokenizer_limit) is int and 0 < tokenizer_limit <= 1_000_000:
        limits.append(tokenizer_limit)
    return min(limits) if limits else fallback


def prepare_training_row(tokenizer, prompt: str, completion: str, limit: int) -> dict:
    """Mask only prompt tokens; never drop target or EOS to fit context."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token")
    prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if not prefix or not target or len(prefix) + len(target) > limit:
        raise ValueError(f"A training example exceeds the {limit}-token model/training context budget")
    return {
        "input_ids": prefix + target,
        "attention_mask": [1] * (len(prefix) + len(target)),
        "labels": [-100] * len(prefix) + target,
    }


def validate_tokenizer(tokenizer, model) -> None:
    size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > size or type(tokenizer.eos_token_id) is not int or not 0 <= tokenizer.eos_token_id < size:
        raise ValueError("Tokenizer vocabulary/EOS is incompatible with model embeddings")
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("This workflow requires a decoder-only causal language model")


def _checkpoint(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("--model must be an existing local Hugging Face checkpoint directory")
    return str(resolved)


def train(model_path: str, output: str, count: int, steps: int, seed: int, suite: str = "compositional") -> dict:
    model_path = _checkpoint(model_path)
    if type(steps) is not int or not 1 <= steps <= 10_000:
        raise ValueError("Training steps must be in 1..10000")
    if suite not in {"primitive", "compositional"}:
        raise ValueError("Unknown dataset suite")
    cases = (generate_composition_dataset if suite == "compositional" else generate_dataset)("train", count, seed)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -e '.[ml]'") from exc
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    validate_tokenizer(tokenizer, model)
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    )
    training_context = min(1024, context_limit(model.config, tokenizer=tokenizer))
    rows = []
    for case in cases:
        rows.append(
            prepare_training_row(
                tokenizer, format_prompt(case), json.dumps(list(case.expected_calls)), training_context
            )
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        max_steps=steps,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        seed=seed,
        report_to=[],
        save_strategy="no",
        logging_steps=max(1, steps // 5),
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    dataset_payload = json.dumps([case.to_dict() for case in cases], sort_keys=True).encode()
    report = {
        "workflow": "actual local LoRA training",
        "model_path": model_path,
        "training_examples": count,
        "suite": suite,
        "dataset_sha256": hashlib.sha256(dataset_payload).hexdigest(),
        "training_context": training_context,
        "completion_only_labels": True,
        "steps": steps,
        "seed": seed,
        "metrics": result.metrics,
        "evaluation": "not run; invoke hf-evaluate separately",
    }
    (destination / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def infer(model_path: str, adapter: str | None, cases: list[Case], max_new_tokens: int) -> list[str]:
    model_path = _checkpoint(model_path)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
        raise ValueError("Token budget must be in 1..1024")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers.generation.utils import GenerationMixin
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -e '.[ml]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, _checkpoint(adapter), local_files_only=True)
    validate_tokenizer(tokenizer, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    outputs = []
    for case in cases:
        inputs = tokenizer(format_prompt(case), return_tensors="pt", return_token_type_ids=False).to(device)
        if inputs["input_ids"].shape[1] + max_new_tokens > context_limit(model.config, 4096, tokenizer):
            raise ValueError("Prompt plus generation exceeds model context")
        with torch.no_grad():
            generation = GenerationConfig(
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            defaults = (
                {"use_model_defaults": False}
                if "use_model_defaults" in inspect.signature(GenerationMixin.generate).parameters
                else {}
            )
            tokens = model.generate(**inputs, generation_config=generation, **defaults)
        suffix = tokens[0, inputs["input_ids"].shape[1] :].tolist()
        if suffix and suffix[-1] == tokenizer.eos_token_id:
            suffix = suffix[:-1]
        outputs.append(tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return outputs
