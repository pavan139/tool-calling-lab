"""Actual optional DPO math and train/resume/evaluation on random local weights."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tool_calling_lab.robot import format_prompt, generate_composition_dataset
from tool_calling_lab.dpo import dpo_loss, evaluate_dpo, load_manifest, sequence_log_probabilities, train_dpo
from tool_calling_lab.hf import train
from tool_calling_lab.preferences import stable_dpo_loss


@unittest.skipUnless(os.environ.get("RUN_ML_SMOKE") == "1", "Set RUN_ML_SMOKE=1 with the ml extra installed")
class DpoIntegrationTests(unittest.TestCase):
    def test_completion_log_probability_shift_padding_and_gradient(self):
        import torch

        probabilities = torch.tensor([[[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]]])
        logits = probabilities.log().requires_grad_()
        model = lambda **_kwargs: SimpleNamespace(logits=logits)
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 2, 1]]),
        }
        actual = sequence_log_probabilities(model, batch)
        self.assertAlmostEqual(float(actual.detach()[0]), math.log(0.1) + math.log(0.5), places=6)
        actual.sum().backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0)
        batch["attention_mask"][0, 3] = 0
        self.assertAlmostEqual(float(sequence_log_probabilities(model, batch).detach()[0]), math.log(0.1), places=6)

    def test_tensor_loss_matches_independent_scalar_and_reference_is_detached(self):
        import torch

        pc, pr, rc, rr = [
            torch.tensor([value], dtype=torch.float64, requires_grad=True) for value in (-4.0, -5.0, -4.0, -5.0)
        ]
        loss = dpo_loss(pc, pr, rc, rr, beta=0.3)
        self.assertAlmostEqual(float(loss.detach()), stable_dpo_loss(-4, -5, -4, -5, 0.3), places=12)
        loss.backward()
        self.assertAlmostEqual(float(pc.grad), -0.15, places=12)
        self.assertAlmostEqual(float(pr.grad), 0.15, places=12)
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    @staticmethod
    def make_checkpoint(path):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        torch.manual_seed(11)
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
        path.mkdir()
        fast.save_pretrained(path)
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
        model.save_pretrained(path)

    def test_actual_dpo_resume_matches_uninterrupted_and_rejects_mismatch(self):
        import torch
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            self.make_checkpoint(base)
            full = train_dpo(str(base), str(root / "full"), count=4, steps=2, checkpoint_every=1)
            first = train_dpo(str(base), str(root / "resume"), count=4, steps=1, checkpoint_every=1)
            resumed = train_dpo(
                str(base), str(root / "resume"), count=4, steps=2, checkpoint_every=1, resume=str(root / "resume")
            )
            full_weights = load_file(str(Path(full["adapter_path"]) / "adapter_model.safetensors"))
            resumed_weights = load_file(str(Path(resumed["adapter_path"]) / "adapter_model.safetensors"))
            self.assertTrue(all(torch.equal(full_weights[key], resumed_weights[key]) for key in full_weights))
            self.assertTrue(any(value.abs().sum() > 0 for key, value in full_weights.items() if "lora_B" in key))
            self.assertGreater(first["history"][0]["gradient_norm"], 0)
            self.assertEqual(full["history"], resumed["history"])
            self.assertEqual(resumed["reference_parameters_with_grad"], 0)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                train_dpo(str(base), str(root / "resume"), count=4, steps=3, beta=0.2, resume=str(root / "resume"))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                train_dpo(str(base), str(root / "resume"), count=5, steps=3, resume=str(root / "resume"))
            evaluation = evaluate_dpo(str(base), str(root / "resume"), count=2, max_new_tokens=4)
            self.assertEqual(evaluation["count"], 2)
            self.assertEqual(evaluation["execution"]["count"], 2)
            self.assertEqual(evaluation["split"], "validation")
            self.assertEqual(load_manifest(root / "resume")[1]["step"], 2)

    def test_sft_adapter_can_be_the_frozen_dpo_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            self.make_checkpoint(base)
            train(str(base), str(root / "sft"), 2, 1, 17)
            report = train_dpo(str(base), str(root / "dpo"), count=2, steps=1, initial_adapter=str(root / "sft"))
            self.assertIsNotNone(report["config"]["reference_adapter_sha256"])
            result = evaluate_dpo(str(base), str(root / "dpo"), count=1, max_new_tokens=2)
            self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main()
