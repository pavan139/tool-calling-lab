"""Reproducible, bounded random-model DPO experiment; no pretrained downloads."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_calling_lab.robot import format_prompt, generate_composition_dataset
from tool_calling_lab.dpo import evaluate_dpo, train_dpo


def run():
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / "base"
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        corpus = []
        for case in generate_composition_dataset("train", 36):
            corpus.extend((format_prompt(case), json.dumps(list(case.expected_calls))))
        tokenizer.train_from_iterator(
            corpus,
            trainers.BpeTrainer(
                vocab_size=512,
                special_tokens=["<eos>", "<unk>", "<bos>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<eos>",
            unk_token="<unk>",
            bos_token="<bos>",
            pad_token="<eos>",
            model_max_length=2048,
        )
        base.mkdir()
        fast.save_pretrained(base)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(fast),
                n_positions=2048,
                n_ctx=2048,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=fast.bos_token_id,
                eos_token_id=fast.eos_token_id,
            )
        )
        model.save_pretrained(base)
        full = train_dpo(str(base), str(root / "full"), count=4, steps=2, checkpoint_every=1)
        train_dpo(str(base), str(root / "resumed"), count=4, steps=1, checkpoint_every=1)
        resumed = train_dpo(
            str(base), str(root / "resumed"), count=4, steps=2, checkpoint_every=1, resume=str(root / "resumed")
        )
        left = load_file(str(Path(full["adapter_path"]) / "adapter_model.safetensors"))
        right = load_file(str(Path(resumed["adapter_path"]) / "adapter_model.safetensors"))
        equal = all(torch.equal(left[key], right[key]) for key in left)
        if not equal:
            raise AssertionError("Interrupted/resumed optimization differs")
        heldout = evaluate_dpo(str(base), str(root / "resumed"), count=2, max_new_tokens=4)
        return {
            "scope": "Actual tiny random-model DPO integration, NOT pretrained quality or meaningful task accuracy",
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "device": full["config"]["device"],
            "model": {"layers": 1, "embedding_size": 16, "heads": 2, "vocabulary": len(fast)},
            "training_pairs": 4,
            "optimizer_steps": 2,
            "loss_history": full["history"],
            "bitwise_resumed_adapter_match": equal,
            "resumed_history_match": full["history"] == resumed["history"],
            "reference_parameters_with_grad": resumed["reference_parameters_with_grad"],
            "training_dataset_sha256": full["config"]["dataset_sha256"],
            "heldout_split": heldout["split"],
            "heldout_cases": heldout["count"],
            "heldout_preference_accuracy": heldout["preference_accuracy"],
            "heldout_execution_metrics": heldout["execution"]["metrics"],
            "generation_budget": heldout["max_new_tokens"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(run(), indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
