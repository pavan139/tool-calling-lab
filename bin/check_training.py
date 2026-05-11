#!/usr/bin/env python3
"""Run Small-Model Tool-Calling Lab checks and optionally validate an installation outside the checkout.

No providers or model downloads are used. --wheel enables the installation checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import venv

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "project.json").is_file())


def strict_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique_keys(pairs: list[tuple]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    if path.stat().st_size > 20_000_000:
        raise ValueError("Report exceeds 20 MB")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected a nonempty JSON report object")
    return value


def repository_layout(root: Path = ROOT) -> dict:
    return strict_json(root / "project.json")["layout"]


def python_paths() -> list[str]:
    plan = repository_layout()
    folders = [*plan["source_roots"], plan["tests_root"], plan["tools_root"]]
    return list(dict.fromkeys(folder for folder in folders if (ROOT / folder).is_dir()))


def source_hash(root: Path = ROOT) -> str:
    plan = repository_layout(root)
    folders = {*plan["source_roots"], plan["tests_root"], plan["tools_root"], plan["config_root"]}
    extensions = {".py", ".go", ".sql", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".csv", ".txt"}
    files = {
        path
        for folder in folders
        for path in (root / folder).rglob("*")
        if path.is_file()
        and path.suffix in extensions
        and not {"__pycache__", ".ruff_cache", "build", "dist"}.intersection(path.relative_to(root).parts)
        and not any(part.endswith((".egg-info", ".dist-info")) for part in path.relative_to(root).parts)
    }
    files.update(
        root / name
        for name in ("pyproject.toml", "project.json", "ruff.toml", "requirements-dev.txt", "go.mod", "go.sum")
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def run_step(name: str, command: list[str], cwd: Path, environment: dict, timeout: int) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, timeout=timeout)
            report = {"name": name, "passed": result.returncode == 0, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            report = {"name": name, "passed": False, "error": "timeout", "timeout_seconds": timeout}
        except OSError as error:
            report = {"name": name, "passed": False, "error": "start_failed", "detail": str(error)}
        for label, handle in (("stdout", stdout), ("stderr", stderr)):
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 32000))
            report[label] = handle.read().decode("utf-8", errors="replace")
            report[f"{label}_truncated"] = size > 32000
    report.update(
        {
            "command": [str(value).replace(str(ROOT), ".") for value in command],
            "seconds": round(time.perf_counter() - started, 6),
        }
    )
    return report


def metadata_check(package: str) -> dict:
    errors = []
    project = strict_json(ROOT / "project.json")
    plan = repository_layout()
    documents = plan["documents"]
    required = ["README.md", "LICENSE", "project.json", *documents.values(), plan["verifier"]]
    if project.get("language", "python") == "python":
        source = plan["source_roots"][0]
        required += ["pyproject.toml", f"{source}/{package}/__init__.py", f"{source}/{package}/__main__.py"]
    else:
        required += ["go.mod"]
    for name in required:
        candidate = ROOT / name
        if not candidate.resolve().is_relative_to(ROOT.resolve()):
            errors.append(f"Path outside repository: {name}")
        elif not candidate.is_file():
            errors.append(f"Missing {name}")
    if (ROOT / "pyproject.toml").exists():
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for key in ("name", "version", "readme", "requires-python"):
            if key not in metadata.get("project", {}):
                errors.append(f"Missing package metadata: {key}")
    architecture = ROOT / documents["architecture"]
    if architecture.is_file() and "```mermaid" not in architecture.read_text():
        errors.append("Architecture note has no Mermaid diagram")
    if project.get("language", "python") == "python" and not list((ROOT / plan["tests_root"]).rglob("test*.py")):
        errors.append("No behavioral tests found")
    if project.get("language") == "go" and not list(ROOT.rglob("*_test.go")):
        errors.append("No Go behavioral tests found")
    excluded = {".git", ".venv", "venv", "dist", "build", "artifacts", "__pycache__", "node_modules"}
    for directory, directories, files in os.walk(ROOT):
        directories[:] = [name for name in directories if name not in excluded]
        for name in files:
            if not name.endswith(".md"):
                continue
            path = Path(directory) / name
            content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if target.startswith(("https:", "http:", "mailto:", "#")):
                    continue
                relative = target.split("#", 1)[0].strip("<>")
                if relative and not (path.parent / relative).exists():
                    errors.append(f"Broken link in {path.relative_to(ROOT)}: {relative}")
    return {"name": "metadata_and_documentation", "passed": not errors, "errors": errors}


def evaluate_cli(
    executable: Path | str, package: str, cwd: Path, environment: dict, timeout: int, isolated: bool
) -> list[dict]:
    checks = []
    with tempfile.TemporaryDirectory(prefix="cli-reports-") as temporary:
        for operation in ("demo", "evaluate"):
            output = Path(temporary) / f"{operation}.json"
            command = [
                str(executable),
                *(["-I"] if isolated else []),
                "-m",
                package,
                operation,
                "--output",
                str(output),
            ]
            result = run_step(f"{'wheel_' if isolated else ''}{operation}", command, cwd, environment, timeout)
            if result["passed"]:
                try:
                    result["report"] = strict_json(output)
                except (OSError, ValueError) as error:
                    result.update(passed=False, artifact_error=str(error))
            checks.append(result)
    return checks


def wheel_checks(package: str, environment: dict, timeout: int) -> list[dict]:
    checks = []
    with tempfile.TemporaryDirectory(prefix="cold-wheel-") as temporary:
        base = Path(temporary)
        wheels = base / "wheels"
        wheels.mkdir()
        built = run_step(
            "wheel_build",
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                str(wheels),
                str(ROOT),
            ],
            ROOT,
            environment,
            timeout,
        )
        checks.append(built)
        if not built["passed"]:
            return checks
        artifacts = list(wheels.glob("*.whl"))
        if len(artifacts) != 1:
            checks.append({"name": "wheel_count", "passed": False, "count": len(artifacts)})
            return checks
        target = base / "environment"
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(target)
        executable = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cold = environment.copy()
        cold.pop("PYTHONPATH", None)
        installed = run_step(
            "wheel_install",
            [str(executable), "-I", "-m", "pip", "install", "--no-deps", "--no-index", str(artifacts[0])],
            base,
            cold,
            timeout,
        )
        checks.append(installed)
        if not installed["passed"]:
            return checks
        code = f"import {package} as module; import sys; from pathlib import Path; assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); print(module.__file__)"
        checks.append(run_step("wheel_import_origin", [str(executable), "-I", "-c", code], base, cold, timeout))
        checks.extend(evaluate_cli(executable, package, base, cold, timeout, isolated=True))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/verification.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--wheel", action="store_true", help="Build and test a normal installation outside this checkout"
    )
    parser.add_argument("--require-lint", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    project = strict_json(ROOT / "project.json")
    package = project["package"]
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / repository_layout()["source_roots"][0]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    checks = [metadata_check(package)]
    if importlib.util.find_spec("ruff"):
        checks.append(
            run_step(
                "lint",
                [sys.executable, "-m", "ruff", "check", *python_paths()],
                ROOT,
                environment,
                args.timeout,
            )
        )
        checks.append(
            run_step(
                "format",
                [sys.executable, "-m", "ruff", "format", "--check", *python_paths()],
                ROOT,
                environment,
                args.timeout,
            )
        )
    elif args.require_lint:
        checks.append({"name": "lint", "passed": False, "error": "ruff_missing_install_requirements_dev"})
    else:
        checks.append({"name": "lint", "passed": True, "status": "not_run", "reason": "ruff_not_installed"})
    checks.append(
        run_step(
            "compile",
            [sys.executable, "-m", "compileall", "-q", *python_paths()],
            ROOT,
            environment,
            args.timeout,
        )
    )
    tests = run_step(
        "unit_tests",
        [sys.executable, "-m", "unittest", "discover", "-s", repository_layout()["tests_root"], "-v"],
        ROOT,
        environment,
        args.timeout,
    )
    summary = tests["stdout"] + tests["stderr"]
    count = re.search(r"Ran (\d+) tests?", summary)
    skipped = re.search(r"skipped=(\d+)", summary)
    test_count = int(count.group(1)) if count else 0
    skip_count = int(skipped.group(1)) if skipped else 0
    if test_count - skip_count < 1:
        tests.update(passed=False, error="no_executed_tests")
    checks.append(tests)
    checks.extend(evaluate_cli(sys.executable, package, ROOT, environment, args.timeout, isolated=False))
    if args.wheel:
        checks.extend(wheel_checks(package, environment, args.timeout))
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": project["repository_name"],
        "source_sha256": source_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "software_contracts_and_labeled_synthetic_experiments",
        "tests_discovered": test_count,
        "tests_skipped": skip_count,
        "tests_executed": test_count - skip_count,
        "external_provider_calls": False,
        "full_model_accuracy_benchmark_run": False,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for check in checks:
        print(f"{check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        if not check["passed"]:
            print(json.dumps(check, indent=2)[:6000])
    print(f"Report: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
