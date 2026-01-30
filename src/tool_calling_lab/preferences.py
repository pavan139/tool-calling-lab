"""Execution-derived preference supervision, with no model-based reward judge."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

from .robot import (
    COMPOSITIONS,
    Case,
    ContractError,
    action_signature,
    evaluate,
    format_prompt,
    generate_composition_dataset,
    parse_output,
)


@dataclass(frozen=True)
class Preference:
    case: Case
    chosen: str
    rejected: str
    chosen_reward: int
    rejected_reward: int
    rejection_kind: str

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "prompt": format_prompt(self.case),
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
            "rejection_kind": self.rejection_kind,
            "signature": action_signature(self.case),
            "provenance": "synthetic_execution_ranked",
        }


def execution_reward(case: Case, output: str) -> int:
    """Nested contract reward; labels are not supplied by the candidate policy.

    8 = full instruction semantics; 4 = same final state but wrong action trace;
    2 = executable wrong task; 1 = schema-valid execution failure; 0 = invalid.
    """
    row = evaluate([case], [output], "preference-label-oracle")["cases"][0]
    return (
        int(row["schema_valid"])
        + int(row["executable"])
        + 2 * int(row["execution_success"])
        + 4 * int(row["instruction_success"])
    )


def candidates(case: Case) -> list[tuple[str, str]]:
    golden = list(case.expected_calls)
    result = []
    for kind in ("wrong_distance", "wrong_direction", "wrong_item", "omit_action", "reverse_order", "extra_move"):
        calls = deepcopy(golden)
        if kind == "wrong_distance":
            first = next(call for call in calls if call["tool"] == "move")
            first["arguments"]["steps"] = first["arguments"]["steps"] % 3 + 1
        elif kind == "wrong_direction":
            first = next(call for call in calls if call["tool"] == "move")
            opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}
            first["arguments"]["direction"] = opposite[first["arguments"]["direction"]]
        elif kind == "wrong_item":
            item_calls = [call for call in calls if call["tool"] in {"pick", "drop"}]
            if not item_calls:
                continue
            item_calls[0]["arguments"]["item"] = "missing_item"
        elif kind == "omit_action":
            if len(calls) == 1:
                continue
            calls.pop()
        elif kind == "reverse_order":
            calls.reverse()
        else:
            calls.append({"tool": "move", "arguments": {"direction": "east", "steps": 1}})
        result.append((kind, json.dumps(calls)))
    result.extend((("invalid_schema", '{"tool":"shell","arguments":{}}'), ("invalid_json", "not-json")))
    return result


def generate_preferences(split: str = "train", count: int = 36, seed: int = 17) -> list[Preference]:
    pairs = []
    for index, case in enumerate(generate_composition_dataset(split, count, seed)):
        chosen_calls = deepcopy(list(case.expected_calls))
        # Positive variants preserve action semantics without exact-call identity.
        for position, call in enumerate(chosen_calls):
            if index % 2 and call["tool"] == "move" and call["arguments"]["steps"] > 1 and len(chosen_calls) < 8:
                signature = "".join({"move": "M", "pick": "P", "drop": "D"}[value["tool"]] for value in chosen_calls)
                if signature[:position] + "M" + signature[position:] not in COMPOSITIONS[split]:
                    continue
                first = deepcopy(call)
                first["arguments"]["steps"] = 1
                call["arguments"]["steps"] -= 1
                chosen_calls.insert(position, first)
                break
        chosen = json.dumps(chosen_calls)
        positive_reward = execution_reward(case, chosen)
        if positive_reward != 8:
            raise ValueError("Preference generator produced an invalid positive")
        ranked = [(kind, text, execution_reward(case, text)) for kind, text in candidates(case)]
        # Alternate hard semantic negatives and format failures; never retain ties.
        eligible = []
        for row in ranked:
            if row[2] >= positive_reward:
                continue
            try:
                signature = "".join(
                    {"move": "M", "pick": "P", "drop": "D"}[value["tool"]] for value in parse_output(row[1])
                )
            except ContractError:
                signature = None
            if signature is None or signature in COMPOSITIONS[split]:
                eligible.append(row)
        eligible.sort(key=lambda row: (-row[2], row[0]))
        kind, rejected, reward = eligible[index % len(eligible)]
        pairs.append(Preference(case, chosen, rejected, positive_reward, reward, kind))
    return pairs


def preferences_hash(pairs: list[Preference]) -> str:
    return hashlib.sha256(json.dumps([row.to_dict() for row in pairs], sort_keys=True).encode()).hexdigest()


def preference_report(split: str = "train", count: int = 36, seed: int = 17) -> dict:
    pairs = generate_preferences(split, count, seed)
    return {
        "schema_version": 1,
        "provenance": "synthetic_execution_ranked",
        "model_generated": False,
        "split": split,
        "seed": seed,
        "count": len(pairs),
        "dataset_sha256": preferences_hash(pairs),
        "action_signatures": sorted({action_signature(row.case) for row in pairs}),
        "reward_counts": {str(reward): sum(row.rejected_reward == reward for row in pairs) for reward in (0, 1, 2, 4)},
        "pairs": [row.to_dict() for row in pairs],
    }


def stable_dpo_loss(
    policy_chosen: float, policy_rejected: float, reference_chosen: float, reference_rejected: float, beta: float = 0.1
) -> float:
    """Scalar oracle used to independently test the differentiable tensor loss."""
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    if not all(math.isfinite(value) for value in values) or beta <= 0:
        raise ValueError("Finite log probabilities and positive beta are required")
    value = -beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
