"""Direct preference optimization with explicit log-probabilities and durable state.

No ML dependency is imported until a tensor/training function is called. The
default preference generator remains fully deterministic and dependency-free.
"""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

from .robot import evaluate, format_prompt
from .hf import _checkpoint, context_limit, prepare_training_row, validate_tokenizer
from .preferences import generate_preferences, preferences_hash


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    import torch
    import torch.nn.functional as functional

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(value.shape != policy_chosen.shape for value in values) or policy_chosen.numel() == 0:
        raise ValueError("Log-probability tensors must be aligned and nonempty")
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite sequence log probabilities")
    margin = beta * ((policy_chosen - policy_rejected) - (reference_chosen.detach() - reference_rejected.detach()))
    return functional.softplus(-margin).mean()


def sequence_log_probabilities(model, batch):
    """Sum completion-only next-token log probabilities, including target EOS.

    Labels at token position j are scored by logits at j-1. Prompt and right
    padding labels are -100; attention masking is enforced independently.
    """
    import torch

    ids, attention, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if ids.ndim != 2 or ids.shape != attention.shape or ids.shape != labels.shape or ids.shape[1] < 2:
        raise ValueError("Aligned rank-two token/mask/label tensors are required")
    targets = labels[:, 1:]
    mask = (targets != -100) & attention[:, 1:].bool()
    if not mask.any(dim=1).all():
        raise ValueError("Each sequence requires at least one scored completion token")
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1, :].float()
    safe_targets = torch.where(mask, targets, 0)
    if (safe_targets < 0).any() or (safe_targets >= logits.shape[-1]).any():
        raise ValueError("Completion label outside output vocabulary")
    selected = torch.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, selected, 0.0).sum(dim=-1)


def pair_batch(tokenizer, pair, limit, device):
    import torch

    rows = [
        prepare_training_row(tokenizer, format_prompt(pair.case), output, limit)
        for output in (pair.chosen, pair.rejected)
    ]
    width = max(len(row["input_ids"]) for row in rows)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = width - len(row["input_ids"])
        result["input_ids"].append(row["input_ids"] + [tokenizer.eos_token_id] * padding)
        result["attention_mask"].append(row["attention_mask"] + [0] * padding)
        result["labels"].append(row["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in result.items()}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    root = Path(path).resolve()
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".vocab", ".merges"}
    files = {
        str(file.relative_to(root)): file_hash(file)
        for file in sorted(root.rglob("*"))
        if file.is_file() and file.suffix in suffixes and ".git" not in file.relative_to(root).parts
    }
    if not files:
        raise ValueError("No model/tokenizer files found in checkpoint")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json_atomic(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_checkpoint(output, policy, tokenizer, optimizer, step, config, history, sample_ids, reference_adapter):
    import torch

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{step:06d}"
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing committed checkpoint")
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        policy.save_pretrained(staging / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(staging / "tokenizer")
        state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "history": history,
            "sample_ids": sample_ids,
            "config_sha256": configuration_hash(config),
        }
        with (staging / "state.pt").open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        files = {
            str(file.relative_to(staging)): file_hash(file) for file in sorted(staging.rglob("*")) if file.is_file()
        }
        manifest = {
            "schema_version": 1,
            "step": step,
            "config": config,
            "config_sha256": configuration_hash(config),
            "files": files,
            "reference_adapter_path": reference_adapter,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for file in staging.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        for directory in [*sorted((value for value in staging.rglob("*") if value.is_dir()), reverse=True), staging]:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.replace(staging, destination)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _write_json_atomic(root / "latest.json", {"checkpoint": destination.name, "step": step})
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_manifest(path):
    root = Path(path).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        candidates = sorted(item for item in root.glob("step-[0-9]*") if (item / "manifest.json").is_file())
        if not candidates:
            raise ValueError("No committed DPO checkpoint found")
        # Directory rename is the commit point; a stale pointer cannot hide a
        # newer committed generation. Unrenamed .pending directories are ignored.
        root = candidates[-1]
    if (root / "manifest.json").stat().st_size > 1024 * 1024:
        raise ValueError("Checkpoint manifest exceeds size limit")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or configuration_hash(manifest["config"]) != manifest.get("config_sha256"):
        raise ValueError("Invalid checkpoint configuration manifest")
    if not isinstance(manifest.get("files"), dict) or "state.pt" not in manifest["files"]:
        raise ValueError("Incomplete checkpoint file manifest")
    for relative, expected in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or file_hash(target) != expected:
            raise ValueError("Checkpoint file integrity check failed")
    return root, manifest


def _load_models(model_path, initial_adapter=None, policy_adapter=None, seed=17):
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the ml extra to run actual DPO") from exc
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    policy = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    validate_tokenizer(tokenizer, policy)
    validate_tokenizer(tokenizer, reference)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    source = policy_adapter or initial_adapter
    if source:
        policy = PeftModel.from_pretrained(policy, str(source), is_trainable=True, local_files_only=True)
    else:
        policy = get_peft_model(
            policy, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
        )
    if initial_adapter:
        reference = PeftModel.from_pretrained(
            reference, str(initial_adapter), is_trainable=False, local_files_only=True
        )
    reference.requires_grad_(False)
    # eval() disables all dropout but does not disable policy autograd. This
    # makes the preference ratio and CPU resume comparison deterministic.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    reference.to(device).eval()
    return policy, reference, tokenizer, device


def train_dpo(
    model_path,
    output,
    count=36,
    steps=20,
    seed=17,
    beta=0.1,
    learning_rate=1e-4,
    checkpoint_every=5,
    resume=None,
    initial_adapter=None,
):
    import torch

    model_path = _checkpoint(model_path)
    if (
        type(steps) is not int
        or not 1 <= steps <= 10_000
        or type(checkpoint_every) is not int
        or not 1 <= checkpoint_every <= 10_000
    ):
        raise ValueError("steps and checkpoint_every must be integer values in 1..10000")
    if type(seed) is not int or not all(math.isfinite(value) and value > 0 for value in (beta, learning_rate)):
        raise ValueError("Integer seed and finite positive beta/learning rate required")
    destination = Path(output).expanduser().resolve()
    base = Path(model_path)
    if destination.is_relative_to(base) or base.is_relative_to(destination):
        raise ValueError("DPO output and base checkpoint must be separate directory trees")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise ValueError("Nonempty output requires --resume; existing runs are never overwritten")
    prior_path, prior_manifest = load_manifest(resume) if resume else (None, None)
    if prior_manifest and initial_adapter is None:
        initial_adapter = prior_manifest.get("reference_adapter_path")
    initial_adapter = _checkpoint(initial_adapter) if initial_adapter else None
    if initial_adapter and (
        destination.is_relative_to(Path(initial_adapter)) or Path(initial_adapter).is_relative_to(destination)
    ):
        raise ValueError("DPO output and reference adapter must be separate directory trees")
    pairs = generate_preferences("train", count, seed)
    policy, reference, tokenizer, device = _load_models(
        model_path, initial_adapter, prior_path / "adapter" if prior_path else None, seed
    )
    limit = min(
        1024, context_limit(policy.config, tokenizer=tokenizer), context_limit(reference.config, tokenizer=tokenizer)
    )
    # Validate all rows before taking any optimizer step.
    for pair in pairs:
        pair_batch(tokenizer, pair, limit, device)
    config = {
        "algorithm": "completion-only-DPO-LoRA/v1",
        "base_sha256": checkpoint_identity(model_path),
        "reference_adapter_sha256": checkpoint_identity(initial_adapter) if initial_adapter else None,
        "dataset_sha256": preferences_hash(pairs),
        "count": count,
        "seed": seed,
        "suite": "compositional",
        "beta": beta,
        "learning_rate": learning_rate,
        "context_limit": limit,
        "device": device,
        "dropout_enabled": False,
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.01,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
    }
    if prior_manifest and prior_manifest["config_sha256"] != configuration_hash(config):
        raise ValueError("Resume rejected: data, model, tokenizer or optimizer configuration changed")
    parameters = [value for value in policy.parameters() if value.requires_grad]
    if not parameters or any(value.requires_grad for value in reference.parameters()):
        raise ValueError("Trainable policy and frozen reference are required")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    completed, history, sample_ids = 0, [], []
    if prior_path:
        state = torch.load(prior_path / "state.pt", map_location="cpu", weights_only=True)
        if state["config_sha256"] != configuration_hash(config) or state["step"] != prior_manifest["step"]:
            raise ValueError("Optimizer state does not match committed manifest")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        completed, history, sample_ids = state["step"], state["history"], state["sample_ids"]
    if completed > steps:
        raise ValueError("Requested total steps precede the resumed step")
    latest = prior_path
    for step in range(completed, steps):
        order = list(range(len(pairs)))
        random.Random(seed + step // len(pairs)).shuffle(order)
        pair = pairs[order[step % len(pairs)]]
        batch = pair_batch(tokenizer, pair, limit, device)
        optimizer.zero_grad(set_to_none=True)
        policy_logps = sequence_log_probabilities(policy, batch)
        with torch.no_grad():
            reference_logps = sequence_log_probabilities(reference, batch)
        loss = dpo_loss(policy_logps[:1], policy_logps[1:], reference_logps[:1], reference_logps[1:], beta)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite DPO loss; no checkpoint committed for failed step")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm),
                "case_id": pair.case.id,
                "rejected_reward": pair.rejected_reward,
            }
        )
        sample_ids.append(pair.case.id)
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            latest = save_checkpoint(
                destination, policy, tokenizer, optimizer, step + 1, config, history, sample_ids, initial_adapter
            )
    report = {
        "workflow": "actual local completion-only DPO",
        "completed_steps": steps,
        "resumed_from_step": completed,
        "training_pairs": len(pairs),
        "config": config,
        "checkpoint": str(latest),
        "adapter_path": str(latest / "adapter"),
        "history": history,
        "reference_parameters_with_grad": sum(value.grad is not None for value in reference.parameters()),
        "scope": "Optimization evidence only; evaluate held-out execution separately.",
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination / "dpo_report.json", report)
    return report


def evaluate_dpo(model_path, checkpoint, split="validation", count=12, seed=17, max_new_tokens=64):
    import torch
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    if split not in {"validation", "test"}:
        raise ValueError("DPO evaluation requires validation or held-out test split")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be an integer in 1..1024")
    model_path = _checkpoint(model_path)
    path, manifest = load_manifest(checkpoint)
    if checkpoint_identity(model_path) != manifest["config"]["base_sha256"]:
        raise ValueError("Evaluation base model differs from the checkpoint reference")
    initial = manifest.get("reference_adapter_path")
    if initial and checkpoint_identity(initial) != manifest["config"]["reference_adapter_sha256"]:
        raise ValueError("Reference adapter differs from training")
    pairs = generate_preferences(split, count, seed)
    policy, reference, tokenizer, device = _load_models(model_path, initial, path / "adapter", seed)
    limit = min(
        1024, context_limit(policy.config, tokenizer=tokenizer), context_limit(reference.config, tokenizer=tokenizer)
    )
    rows, outputs = [], []
    for pair in pairs:
        batch = pair_batch(tokenizer, pair, limit, device)
        with torch.no_grad():
            p = sequence_log_probabilities(policy, batch)
            r = sequence_log_probabilities(reference, batch)
        margin = float(((p[0] - p[1]) - (r[0] - r[1])).cpu())
        rows.append(
            {
                "case_id": pair.case.id,
                "reference_adjusted_margin": margin,
                "preferred": margin > 0,
                "tie": margin == 0,
                "chosen_reward": pair.chosen_reward,
                "rejected_reward": pair.rejected_reward,
            }
        )
        encoded = tokenizer(format_prompt(pair.case), return_tensors="pt", return_token_type_ids=False).to(device)
        if encoded["input_ids"].shape[1] + max_new_tokens > context_limit(policy.config, tokenizer=tokenizer):
            raise ValueError("Evaluation prompt plus generation exceeds model context")
        config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        options = (
            {"use_model_defaults": False}
            if "use_model_defaults" in inspect.signature(GenerationMixin.generate).parameters
            else {}
        )
        with torch.no_grad():
            tokens = policy.generate(**encoded, generation_config=config, **options)
        suffix = tokens[0, encoded["input_ids"].shape[1] :].tolist()
        if suffix and suffix[-1] == tokenizer.eos_token_id:
            suffix.pop()
        outputs.append(tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return {
        "workflow": "actual local DPO checkpoint evaluation",
        "split": split,
        "count": len(rows),
        "dataset_sha256": preferences_hash(pairs),
        "preference_accuracy": sum(row["preferred"] for row in rows) / len(rows),
        "ties": sum(row["tie"] for row in rows),
        "preference_cases": rows,
        "execution": evaluate([pair.case for pair in pairs], outputs, "actual local DPO policy"),
        "max_new_tokens": max_new_tokens,
        "checkpoint": str(path),
        "limitation": "Synthetic preferences and bounded generation; no broad capability claim.",
    }
