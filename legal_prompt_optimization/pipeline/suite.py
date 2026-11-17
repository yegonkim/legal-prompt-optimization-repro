"""Run a repository experiment suite without host-specific launch scripts."""

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import inference_signature, load
from .engine import task_for
from .io import dataset, digest, file_hash, read, validate_contract, write


def freeze_archived(data, config_path, prompt_path, out):
    """Bind explicitly imported instructions to verified data, without optimization."""
    config = load(config_path)
    groups = dataset(data)
    validate_contract(groups, config["data"])
    prompts = read(prompt_path)
    import hashlib

    for role, text in prompts["program"].items():
        if hashlib.sha256(text.encode()).hexdigest() != prompts["sha256"][role]:
            raise ValueError("Archived prompt hash mismatch")
    if config["mode"] != "sequential":
        raise ValueError("Archived sequential prompts require mode: sequential")
    if config["data"]["manifest"] != prompts["dataset"]:
        raise ValueError(
            "Archived evaluation requires the matching frozen dataset contract"
        )
    result = {
        "version": 1,
        "program": prompts["program"],
        "task": asdict(task_for(groups)),
        "selected_on": "archived SELECT; no new optimization",
        "dataset_sha256": file_hash(data),
        "config_sha256": digest(config),
        "inference_signature": inference_signature(config),
        "optimization_ids": [
            r["record_id"] for s in ["fit", "select"] for r in groups[s]
        ],
        "mode": "sequential",
        "imported_prompt_sha256": file_hash(prompt_path),
        "numerical_reproduction_verified": False,
    }
    result["sha256"] = digest(result)
    out = Path(out)
    if out.exists() and read(out) != result:
        raise ValueError("Refusing to replace a different frozen winner")
    write(out, result)
    return result


def commands(suite, root, datasets, stages, output, archived=False, offline=False):
    entries = suite["datasets"]
    unknown = set(datasets) - set(entries)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    for name in datasets:
        entry = entries[name]
        config = root / entry["config"]
        load(config)  # Reject bad configs even during dry run.
        data = output / "data" / (name + ".jsonl")
        run = output / name
        winner = run / (
            "archived/winner.json" if archived else "optimization/winner.json"
        )
        common = ["--data", str(data), "--config", str(config)]
        for stage in stages:
            if stage == "prepare":
                if data.exists():
                    yield [
                        sys.executable,
                        "-m",
                        "legal_prompt_optimization",
                        "validate",
                        *common,
                    ]
                    continue
                cmd = [
                    "prepare",
                    "--dataset",
                    name,
                    "--out",
                    str(data),
                    "--cache-dir",
                    str(output / "sources"),
                ]
                if offline:
                    cmd += ["--offline"]
            elif stage == "optimize":
                if archived:
                    yield [
                        sys.executable,
                        "-m",
                        __name__,
                        "freeze-archived",
                        *common,
                        "--prompts",
                        str(root / entry["prompts"]),
                        "--out",
                        str(winner),
                    ]
                    continue
                cmd = ["optimize", *common, "--out", str(run / "optimization")]
                if (run / "optimization").exists():
                    cmd += ["--resume"]
            else:
                cmd = [
                    "evaluate",
                    *common,
                    "--winner",
                    str(winner),
                    "--split",
                    stage,
                    "--out",
                    str(run / ("archived-" + stage if archived else stage)),
                ]
                if (run / ("archived-" + stage if archived else stage)).exists():
                    cmd += ["--resume"]
            yield [sys.executable, "-m", "legal_prompt_optimization", *cmd]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "freeze-archived":
        p = argparse.ArgumentParser(
            description="Import archived prompts without optimization"
        )
        for k in ["data", "config", "prompts", "out"]:
            p.add_argument("--" + k, required=True, type=Path)
        a = p.parse_args(argv[1:])
        freeze_archived(a.data, a.config, a.prompts, a.out)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=Path("configs/experiments/suite.yaml"))
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["prepare", "optimize", "dev", "test"],
        default=["prepare", "optimize", "dev", "test"],
    )
    p.add_argument("--out", type=Path, default=Path("runs/public-suite"))
    p.add_argument(
        "--archived-prompts",
        action="store_true",
        help="Import saved sequential prompts instead of optimizing",
    )
    p.add_argument(
        "--offline", action="store_true", help="Require cached preparation inputs"
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    spec = yaml.safe_load(a.suite.read_text())
    if set(spec) != {"version", "root", "datasets"} or spec["version"] != 1:
        raise ValueError("Unsupported suite schema")
    root = (a.suite.parent / spec["root"]).resolve()
    names = list(spec["datasets"]) if a.datasets == ["all"] else a.datasets
    if len(set(a.stages)) != len(a.stages) or a.stages != sorted(
        a.stages, key=["prepare", "optimize", "dev", "test"].index
    ):
        raise ValueError(
            "Stages must be unique and ordered prepare, optimize, dev, test"
        )
    for cmd in commands(
        spec, root, names, a.stages, a.out.resolve(), a.archived_prompts, a.offline
    ):
        print(shlex.join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
