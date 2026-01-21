"""A bounded simulator. Model-produced JSON is always untrusted."""

from __future__ import annotations

import json
import hashlib
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

MAX_CALLS = 8
MAX_OUTPUT_BYTES = 16_384
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
SPLIT_ITEMS = {
    "train": ("ruby", "sapphire", "coin"),
    "validation": ("orb", "crystal", "token"),
    "test": ("relic", "gem", "scroll"),
}
PHRASES = {
    "train": ("Go {n} cells {d}", "Pick up {item}", "Drop {item}"),
    "validation": ("Travel {d} for {n} squares", "Collect {item}", "Put down {item}"),
    "test": ("Move {n} spaces toward {d}", "Take {item}", "Leave {item}"),
}


class ContractError(ValueError):
    """Invalid tools, environment, output, or execution."""


@dataclass(frozen=True)
class State:
    x: int
    y: int
    inventory: tuple[str, ...] = ()
    items: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class World:
    size: int = 7
    obstacles: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    instruction: str
    initial: State
    world: World
    expected_calls: tuple[dict[str, Any], ...]
    expected_state: State
    suite: str = "primitive"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,23}", value) is not None


def validate_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CALLS:
        raise ContractError(f"Expected 1..{MAX_CALLS} calls")
    for call in value:
        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
            raise ContractError("Each call needs exactly tool and arguments")
        tool, args = call["tool"], call["arguments"]
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise ContractError("Tool must be a string and arguments an object")
        if tool == "move":
            if set(args) != {"direction", "steps"}:
                raise ContractError("move needs direction and steps")
            if not isinstance(args["direction"], str) or args["direction"] not in DIRECTIONS:
                raise ContractError("Unknown direction")
            if type(args["steps"]) is not int or not 1 <= args["steps"] <= 3:
                raise ContractError("steps must be an integer in 1..3")
        elif tool in {"pick", "drop"}:
            if set(args) != {"item"} or not _name(args["item"]):
                raise ContractError("pick/drop need one bounded item name")
        else:
            raise ContractError("Tool not allowed")
    return value


def parse_output(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise ContractError("Output must be text")
    try:
        output_bytes = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError("Output must be valid UTF-8 text") from exc
    if len(output_bytes) > MAX_OUTPUT_BYTES:
        raise ContractError("Output exceeds byte budget")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("Duplicate JSON keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, RecursionError) as exc:
        raise ContractError("Expected one JSON array, with no prose") from exc
    return validate_calls(decoded)


def execute(world: World, initial: State, calls: list[dict[str, Any]]) -> State:
    validate_calls(calls)
    if not isinstance(world, World) or not isinstance(initial, State):
        raise ContractError("Expected typed World and State records")
    if type(world.size) is not int or not 1 <= world.size <= 64:
        raise ContractError("World size must be 1..64")

    def inside(x: int, y: int) -> bool:
        return type(x) is int and type(y) is int and 0 <= x < world.size and 0 <= y < world.size

    if not inside(initial.x, initial.y):
        raise ContractError("Initial position outside world")
    if not isinstance(world.obstacles, tuple) or len(world.obstacles) > 4096:
        raise ContractError("Obstacles must be a bounded tuple")
    for point in world.obstacles:
        if not isinstance(point, tuple) or len(point) != 2 or not inside(*point):
            raise ContractError("Invalid obstacle")
    if (initial.x, initial.y) in world.obstacles:
        raise ContractError("Robot starts inside obstacle")
    if (
        not isinstance(initial.inventory, tuple)
        or not isinstance(initial.items, tuple)
        or any(not isinstance(item, tuple) or len(item) != 3 for item in initial.items)
    ):
        raise ContractError("State inventory and items must be typed tuples")
    if len(initial.inventory) + len(initial.items) > 128:
        raise ContractError("At most 128 items are supported")
    names = list(initial.inventory) + [item[0] for item in initial.items]
    if any(not _name(name) for name in names) or len(names) != len(set(names)):
        raise ContractError("Item names must be valid and unique")
    if any(not inside(x, y) or (x, y) in world.obstacles for _, x, y in initial.items):
        raise ContractError("Invalid item location")
    x, y = initial.x, initial.y
    inventory = set(initial.inventory)
    items = {name: (ix, iy) for name, ix, iy in initial.items}
    for call in calls:
        args = call["arguments"]
        if call["tool"] == "move":
            dx, dy = DIRECTIONS[args["direction"]]
            for _ in range(args["steps"]):
                nx, ny = x + dx, y + dy
                if not inside(nx, ny) or (nx, ny) in world.obstacles:
                    raise ContractError("Movement leaves world or crosses obstacle")
                x, y = nx, ny
        elif call["tool"] == "pick":
            item = args["item"]
            if items.get(item) != (x, y):
                raise ContractError("Item is not at robot position")
            inventory.add(item)
            del items[item]
        else:
            item = args["item"]
            if item not in inventory:
                raise ContractError("Cannot drop unowned item")
            inventory.remove(item)
            items[item] = (x, y)
    return State(x, y, tuple(sorted(inventory)), tuple(sorted((name, *point) for name, point in items.items())))


def generate_dataset(split: str = "test", count: int = 24, seed: int = 17) -> list[Case]:
    if split not in PHRASES or type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("Unknown split or count outside 1..10000")
    # Split-specific templates AND item vocabularies; generation never reads another split.
    rng = random.Random(f"{seed}:{split}")
    cases: list[Case] = []
    move_phrase, pick_phrase, drop_phrase = PHRASES[split]
    for index in range(count):
        direction = tuple(DIRECTIONS)[index % 4]
        steps = 1 + (index // 4) % 3
        action_kind = (index // 12) % 3
        item = rng.choice(SPLIT_ITEMS[split])
        dx, dy = DIRECTIONS[direction]
        x = rng.randrange(max(0, -dx * steps), min(7, 7 - dx * steps))
        y = rng.randrange(max(0, -dy * steps), min(7, 7 - dy * steps))
        calls: list[dict[str, Any]] = [{"tool": "move", "arguments": {"direction": direction, "steps": steps}}]
        sentences = [move_phrase.format(n=steps, d=direction)]
        items: tuple[tuple[str, int, int], ...] = ()
        if action_kind >= 1:
            items = ((item, x + dx * steps, y + dy * steps),)
            calls.append({"tool": "pick", "arguments": {"item": item}})
            sentences.append(pick_phrase.format(item=item))
        if action_kind >= 2:
            opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}[direction]
            calls.append({"tool": "move", "arguments": {"direction": opposite, "steps": 1}})
            sentences.append(move_phrase.format(n=1, d=opposite))
            calls.append({"tool": "drop", "arguments": {"item": item}})
            sentences.append(drop_phrase.format(item=item))
        # Final-state labels are computed directly from task construction, not execute().
        final_x, final_y = x + dx * steps, y + dy * steps
        final_inventory = (item,) if action_kind == 1 else ()
        final_items: tuple[tuple[str, int, int], ...] = ()
        if action_kind == 2:
            final_x, final_y = final_x - dx, final_y - dy
            final_items = ((item, final_x, final_y),)
        goal = State(final_x, final_y, final_inventory, final_items)
        cases.append(
            Case(
                f"{split}-{seed}-{index:05}",
                split,
                ". ".join(sentences) + ".",
                State(x, y, items=items),
                World(),
                tuple(calls),
                goal,
            )
        )
    return cases


COMPOSITIONS = {
    "train": ("M", "MP", "MPMD"),
    "validation": ("MMP", "MPM"),
    "test": ("MMPMD", "MPMMD", "MMPMMD"),
}


def generate_composition_dataset(split: str = "test", count: int = 36, seed: int = 17) -> list[Case]:
    """Hold out operation compositions as well as phrases and item vocabularies.

    A direct construction oracle tracks goal coordinates/items without invoking
    the tested executor. No expected state or calls appear in model prompts.
    """
    if split not in COMPOSITIONS or type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("Unknown split or count outside 1..10000")
    rng = random.Random(f"composition:{seed}:{split}")
    move_phrase, pick_phrase, drop_phrase = PHRASES[split]
    result = []
    for index in range(count):
        signature = COMPOSITIONS[split][index % len(COMPOSITIONS[split])]
        x, y = rng.randrange(7), rng.randrange(7)
        start = (x, y)
        item = rng.choice(SPLIT_ITEMS[split])
        calls, sentences = [], []
        initial_items: tuple[tuple[str, int, int], ...] = ()
        inventory: tuple[str, ...] = ()
        final_items: tuple[tuple[str, int, int], ...] = ()
        for symbol in signature:
            if symbol == "M":
                choices = [
                    (direction, steps)
                    for direction, (dx, dy) in DIRECTIONS.items()
                    for steps in (1, 2, 3)
                    if 0 <= x + dx * steps < 7 and 0 <= y + dy * steps < 7
                ]
                direction, steps = rng.choice(choices)
                dx, dy = DIRECTIONS[direction]
                x, y = x + dx * steps, y + dy * steps
                calls.append({"tool": "move", "arguments": {"direction": direction, "steps": steps}})
                sentences.append(move_phrase.format(n=steps, d=direction))
            elif symbol == "P":
                initial_items = ((item, x, y),)
                inventory = (item,)
                calls.append({"tool": "pick", "arguments": {"item": item}})
                sentences.append(pick_phrase.format(item=item))
            else:
                inventory = ()
                final_items = ((item, x, y),)
                calls.append({"tool": "drop", "arguments": {"item": item}})
                sentences.append(drop_phrase.format(item=item))
        goal = State(x, y, inventory, final_items)
        result.append(
            Case(
                f"composition-{split}-{seed}-{index:05}",
                split,
                ". ".join(sentences) + ".",
                State(*start, items=initial_items),
                World(),
                tuple(calls),
                goal,
                "compositional",
            )
        )
    return result


def action_signature(case: Case) -> str:
    return "".join({"move": "M", "pick": "P", "drop": "D"}[call["tool"]] for call in case.expected_calls)


def canonical_actions(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Comparable action semantics, allowing moves to be split into unit steps."""
    result = []
    for call in validate_calls(calls):
        args = call["arguments"]
        if call["tool"] == "move":
            result.extend([("move", args["direction"])] * args["steps"])
        else:
            result.append((call["tool"], args["item"]))
    return result


def reference_compile(instruction: str) -> list[dict[str, Any]]:
    """Handwritten controlled-language reference; NOT a learned model."""
    calls: list[dict[str, Any]] = []
    if not isinstance(instruction, str) or len(instruction) > 4096:
        raise ContractError("Instruction too long")
    for sentence in instruction.rstrip(".").split(". "):
        patterns = (
            r"Go (\d+) cells (north|east|south|west)",
            r"Travel (north|east|south|west) for (\d+) squares",
            r"Move (\d+) spaces toward (north|east|south|west)",
        )
        matched = False
        for index, pattern in enumerate(patterns):
            match = re.fullmatch(pattern, sentence)
            if match:
                first, second = match.groups()
                direction, steps = (first, int(second)) if index == 1 else (second, int(first))
                calls.append({"tool": "move", "arguments": {"direction": direction, "steps": steps}})
                matched = True
                break
        if matched:
            continue
        match = re.fullmatch(r"(Pick up|Collect|Take|Drop|Put down|Leave) ([a-z][a-z0-9_]{0,23})", sentence)
        if not match:
            raise ContractError("Unsupported controlled-language instruction")
        verb, item = match.groups()
        calls.append(
            {"tool": "pick" if verb in {"Pick up", "Collect", "Take"} else "drop", "arguments": {"item": item}}
        )
    return validate_calls(calls)


def evaluate(cases: list[Case], outputs: list[str], system: str) -> dict[str, Any]:
    if not cases or len(cases) != len(outputs):
        raise ValueError("Need one output per nonempty case list")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    rows = []
    for case, output in zip(cases, outputs):
        row: dict[str, Any] = {
            "id": case.id,
            "instruction": case.instruction,
            "output": output,
            "schema_valid": False,
            "executable": False,
            "exact_calls": False,
            "execution_success": False,
            "instruction_success": False,
            "action_signature": action_signature(case),
        }
        expected = case.expected_state
        if execute(case.world, case.initial, list(case.expected_calls)) != expected:
            raise ValueError(f"Golden calls disagree with independent goal for {case.id}")
        try:
            calls = parse_output(output)
            row["schema_valid"] = True
            row["exact_calls"] = calls == list(case.expected_calls)
            actual = execute(case.world, case.initial, calls)
            row["executable"] = True
            row["execution_success"] = actual == expected
            row["instruction_success"] = actual == expected and canonical_actions(calls) == canonical_actions(
                list(case.expected_calls)
            )
            row["actual_state"] = asdict(actual)
        except ContractError as exc:
            row["error"] = str(exc)
        row["expected_state"] = asdict(expected)
        rows.append(row)
    payload = json.dumps([case.to_dict() for case in cases], sort_keys=True).encode()
    return {
        "system": system,
        "dataset": "synthetic controlled-language robot tasks",
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "suites": sorted({case.suite for case in cases}),
        "count": len(rows),
        "goal_oracle": "independently constructed final-state labels plus canonical action traces",
        "metrics": {
            key: sum(row[key] for row in rows) / len(rows)
            for key in ("schema_valid", "executable", "exact_calls", "execution_success", "instruction_success")
        },
        "cases": rows,
    }


def format_prompt(case: Case) -> str:
    contract = 'Return ONLY a JSON array of calls. Each call has exactly "tool" and "arguments". Allowed: move(direction: north/east/south/west, steps: integer 1..3), pick(item: string), drop(item: string). At most 8 calls. No prose.'
    return f"{contract}\nWorld: {json.dumps(asdict(case.world))}\nState: {json.dumps(asdict(case.initial))}\nInstruction: {case.instruction}\nJSON:"
