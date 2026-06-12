from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from tool_calling_lab.robot import COMPOSITIONS, State, parse_output
from tool_calling_lab.dpo import configuration_hash, load_manifest
from tool_calling_lab.preferences import execution_reward, generate_preferences, preference_report, stable_dpo_loss


class PreferenceDataTests(unittest.TestCase):
    def test_pairs_are_execution_derived_and_strictly_ordered(self):
        for split in COMPOSITIONS:
            pairs = generate_preferences(split, 18)
            self.assertEqual(pairs, generate_preferences(split, 18))
            for pair in pairs:
                self.assertEqual(pair.chosen_reward, execution_reward(pair.case, pair.chosen))
                self.assertEqual(pair.rejected_reward, execution_reward(pair.case, pair.rejected))
                self.assertEqual(pair.chosen_reward, 8)
                self.assertLess(pair.rejected_reward, pair.chosen_reward)

    def test_successful_targets_preserve_compositional_split_isolation(self):
        signatures = {}
        for split in COMPOSITIONS:
            signatures[split] = set()
            for pair in generate_preferences(split, 24):
                calls = parse_output(pair.chosen)
                signature = "".join({"move": "M", "pick": "P", "drop": "D"}[call["tool"]] for call in calls)
                self.assertIn(signature, COMPOSITIONS[split])
                signatures[split].add(signature)
        self.assertFalse(signatures["train"] & signatures["test"])
        self.assertFalse(signatures["validation"] & signatures["test"])

    def test_reward_cannot_trust_a_corrupted_goal(self):
        pair = generate_preferences(count=1)[0]
        with self.assertRaisesRegex(ValueError, "independent goal"):
            execution_reward(replace(pair.case, expected_state=State(-1, -1)), pair.chosen)

    def test_dataset_contains_hard_and_format_negatives(self):
        report = preference_report(count=36)
        self.assertGreater(report["reward_counts"]["2"], 0)
        self.assertGreater(report["reward_counts"]["0"], 0)
        self.assertFalse(report["model_generated"])
        self.assertEqual(report["count"], len(report["pairs"]))
        self.assertEqual(len(report["dataset_sha256"]), 64)


class DpoScalarTests(unittest.TestCase):
    def test_equal_policy_reference_has_log_two_loss(self):
        self.assertAlmostEqual(stable_dpo_loss(-3, -5, -3, -5), math.log(2))

    def test_preference_direction_and_reference_adjustment(self):
        middle = stable_dpo_loss(-3, -5, -3, -5)
        self.assertLess(stable_dpo_loss(-2, -5, -3, -5), middle)
        self.assertGreater(stable_dpo_loss(-3, -4, -3, -5), middle)
        self.assertAlmostEqual(stable_dpo_loss(-30, -50, -30, -50), middle)

    def test_numerically_stable_extremes(self):
        self.assertTrue(math.isfinite(stable_dpo_loss(-1_000_000, 0, 0, 0, 1)))
        self.assertAlmostEqual(stable_dpo_loss(-1_000_000, 0, 0, 0, 1), 1_000_000)
        self.assertEqual(stable_dpo_loss(0, -1_000_000, 0, 0, 1), 0)
        for beta in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                stable_dpo_loss(-1, -2, -1, -2, beta)


class CheckpointManifestTests(unittest.TestCase):
    @staticmethod
    def commit(root, step):
        path = root / f"step-{step:06d}"
        path.mkdir()
        content = f"state-{step}".encode()
        (path / "state.pt").write_bytes(content)
        config = {"dataset_sha256": "example", "beta": 0.1}
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "step": step,
                    "config": config,
                    "config_sha256": configuration_hash(config),
                    "files": {"state.pt": hashlib.sha256(content).hexdigest()},
                }
            )
        )
        return path

    def test_committed_generation_wins_over_stale_pointer_and_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.commit(root, 1)
            latest = self.commit(root, 2)
            (root / "latest.json").write_text('{"checkpoint":"step-000001"}')
            (root / ".pending-interrupted").mkdir()
            self.assertEqual(load_manifest(root)[0], latest)

    def test_corrupt_state_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.commit(Path(directory), 1)
            (path / "state.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_manifest(path)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["files"] = {"state.pt": hashlib.sha256(b"corrupt").hexdigest(), "../outside": "x"}
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_manifest(path)

    def test_configuration_digest_is_order_invariant_and_value_sensitive(self):
        self.assertEqual(configuration_hash({"x": 1, "y": 2}), configuration_hash({"y": 2, "x": 1}))
        self.assertNotEqual(configuration_hash({"beta": 0.1}), configuration_hash({"beta": 0.2}))


if __name__ == "__main__":
    unittest.main()
