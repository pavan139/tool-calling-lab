"""Opt-in local integration checks; never download pretrained weights."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from tool_calling_lab.robot import format_prompt, generate_dataset
from tool_calling_lab.hf import infer, train


@unittest.skipUnless(os.environ.get("RUN_ML_SMOKE") == "1", "Set RUN_ML_SMOKE=1 with the ml extra installed")
class LocalHuggingFaceSmoke(unittest.TestCase):
    def test_tiny_random_model_training_adapter_loading_and_raw_special_tokens(self):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base"
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            corpus = []
            for split in ("train", "validation", "test"):
                for case in generate_dataset(split, 36):
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
            checkpoint.mkdir()
            fast.save_pretrained(checkpoint)
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
            model.save_pretrained(checkpoint)
            adapter = Path(directory) / "adapter"
            report = train(str(checkpoint), str(adapter), 2, 1, 7, "primitive")
            self.assertEqual(report["steps"], 1)
            self.assertEqual(report["training_examples"], 2)
            self.assertTrue((adapter / "adapter_config.json").is_file())
            self.assertEqual(len(infer(str(checkpoint), str(adapter), generate_dataset("test", 1), 4)), 1)
            with torch.no_grad():
                model.transformer.ln_f.bias.fill_(1.0)
                model.lm_head.weight.zero_()
                model.lm_head.weight[fast.bos_token_id].fill_(1.0)
            model.generation_config.forced_eos_token_id = fast.eos_token_id
            model.generation_config.min_new_tokens = 8
            model.save_pretrained(checkpoint)
            output = infer(str(checkpoint), None, generate_dataset("test", 1), 2)[0]
            self.assertIn("<bos>", output, "Untrusted generated special tokens must not be stripped")


if __name__ == "__main__":
    unittest.main()
