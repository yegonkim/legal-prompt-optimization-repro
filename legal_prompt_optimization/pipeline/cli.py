"""Public entry point. Help and validation do not load models or require a GPU."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULTS, inference_signature, load
from .engine import evaluate, metric, optimize, task_for
from .io import dataset, digest, file_hash, read, validate_contract, write


def start_run(out, plan, resume):
    out = Path(out)
    if out.exists():
        if not resume:
            raise FileExistsError(
                f"{out} exists; use --resume to continue the identical run"
            )
        if not (out / "plan.json").exists() or read(out / "plan.json") != plan:
            raise ValueError(
                "Resume requires identical resolved config, dataset, operation and selected prompts"
            )
    else:
        if resume:
            raise FileNotFoundError("--resume requires an existing run")
        out.mkdir(parents=True)
        write(out / "plan.json", plan)
    write(out / "resolved-config.json", plan["config"])


def table(metrics, configured_metric="auto"):
    aggregation = "vote4" if "vote4" in metrics["baseline"] else "median4"
    names = [
        "Baseline pass@1",
        f"Baseline {aggregation}",
        "Optimized pass@1",
        f"Optimized {aggregation}",
        "Response + judge",
    ]
    if metrics.get("protocol") == "legal_gepa_v1":
        names[0] = "Baseline individual (mean of 16)"
        names[2] = "Optimized individual (mean of 16)"
    scores = [
        metrics["baseline"]["pass1"],
        metrics["baseline"][aggregation],
        metrics["optimized_response"]["pass1"],
        metrics["optimized_response"][aggregation],
        metrics["judge"],
    ]
    metric_name = (
        configured_metric
        if configured_metric != "auto"
        else next(k for k in ["macro_f1", "mae", "brier"] if k in scores[0])
    )
    lines = [
        "| Method | " + metric_name + " | Valid | Failed | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in zip(names, scores):
        value = "undefined" if s[metric_name] is None else f"{s[metric_name]:.6g}"
        lines.append(
            f"| {name} | {value} | {s['valid']} | {s['failed']} | {s['coverage']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None, backend_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "preflight", "optimize", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name != "preflight":
            p.add_argument("--data", type=Path, required=True)
        if name in ["optimize", "evaluate"]:
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--resume", action="store_true")
        if name == "optimize":
            p.add_argument(
                "--response",
                type=Path,
                help="Response winner for judge-only optimization",
            )
        if name == "evaluate":
            p.add_argument("--winner", type=Path, required=True)
            p.add_argument("--split", choices=["dev", "test"], default="dev")
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path(".source-cache"))
    p.add_argument("--offline", action="store_true")
    p.add_argument("--split-policy", choices=["reference", "official", "random"])
    p.add_argument("--source", type=Path)
    p.add_argument("--opinions", type=Path)
    p.add_argument("--judgments-dir", type=Path)
    for k in ["fit", "select", "dev", "test"]:
        p.add_argument("--" + k + "-size", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--tokenizer", default=DEFAULTS["models"]["student"]["model"])
    p.add_argument(
        "--tokenizer-revision", default=DEFAULTS["models"]["student"]["revision"]
    )
    p.add_argument("--workers", type=int, default=4)
    p = commands.add_parser("serve")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--role", choices=["student", "proposer"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--python", default="python")
    p.add_argument("--memory-fraction", type=float, default=0.45)
    p.add_argument("--max-sequences", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--phi-eos-alignment", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .prepare import prepare

        options = vars(args).copy()
        options.pop("command")
        result = prepare(**options)
        print(json.dumps(result))
        return result
    if args.command == "serve":
        from .serve import serve

        return serve(args)
    config = load(args.config)
    if backend_factory is None:
        from .backend import Backend

        backend_factory = Backend
    if args.command == "preflight":
        import tempfile

        with tempfile.TemporaryDirectory() as cache:
            result = backend_factory(config, cache).preflight(include_proposer=True)
        print(json.dumps(result))
        return result
    groups = dataset(args.data)
    validate_contract(groups, config["data"])
    task = task_for(groups)
    optimize_fn, evaluate_fn = optimize, evaluate
    if config["protocol"] == "legal_gepa_v1":
        from . import per_case

        per_case.validate_data(groups, task, config)
        optimize_fn, evaluate_fn = per_case.optimize, per_case.evaluate
    # Check metric compatibility before any requests.
    sample = next(r for group in groups.values() for r in group)
    metric(task, [sample["answer"]], [sample["answer"]], config["task"]["metric"])
    if args.command == "validate":
        result = {
            "valid": True,
            "task": asdict(task),
            "splits": {s: len(v) for s, v in groups.items()},
            "dataset_sha256": file_hash(args.data),
            "config_sha256": digest(config),
        }
        print(json.dumps(result))
        return result
    plan = {
        "version": 1,
        "operation": args.command,
        "config": config,
        "dataset_sha256": file_hash(args.data),
        "task": asdict(task),
    }
    selected = None
    if args.command == "evaluate" or args.command == "optimize" and args.response:
        path = args.winner if args.command == "evaluate" else args.response
        selected = read(path)
        if selected.get("sha256") != digest(
            {k: v for k, v in selected.items() if k != "sha256"}
        ):
            raise ValueError("Winner artifact hash mismatch")
        if selected["dataset_sha256"] != plan["dataset_sha256"]:
            raise ValueError("Winner and dataset differ")
        if selected["inference_signature"] != inference_signature(config):
            raise ValueError(
                "Evaluation/generator inference settings differ from optimization"
            )
        if selected["task"] != json.loads(json.dumps(asdict(task))):
            raise ValueError("Winner task differs")
        plan["winner_sha256"] = selected["sha256"]
    if args.command == "evaluate":
        plan["split"] = args.split
        if not groups[args.split]:
            raise ValueError("Requested evaluation split is empty")
        if set(selected["optimization_ids"]) & {
            r["record_id"] for r in groups[args.split]
        }:
            raise ValueError("Evaluation overlaps optimization IDs")
    if args.command == "optimize":
        if not groups["fit"] or not groups["select"]:
            raise ValueError("Optimization requires nonempty FIT and SELECT")
        if config["mode"] == "judge" and (
            not selected or set(selected["program"]) != {"response"}
        ):
            raise ValueError(
                "Judge mode requires --response with a response-only winner"
            )
        if args.response and config["mode"] != "judge":
            raise ValueError("--response is only valid in judge mode")
    plan = json.loads(json.dumps(plan))
    start_run(args.out, plan, args.resume)
    result_file = args.out / (
        "winner.json" if args.command == "optimize" else "metrics.json"
    )
    if (
        result_file.exists()
        and (args.out / "status.json").exists()
        and read(args.out / "status.json").get("status") == "complete"
    ):
        result = read(result_file)
        print(json.dumps({"status": "complete", "reused": True, "out": str(args.out)}))
        return result
    write(
        args.out / "status.json",
        {"status": "running", "operation": args.command, "started_at": time.time()},
    )
    try:
        backend = backend_factory(config, args.out / "cache")
        backend.preflight(include_proposer=args.command == "optimize")
        if args.command == "optimize":
            program = optimize_fn(
                backend,
                {k: groups[k] for k in ["fit", "select"]},
                task,
                config,
                args.out,
                selected["program"] if selected else None,
            )
            result = {
                "version": 1,
                "program": program,
                "task": asdict(task),
                "selected_on": "SELECT",
                "dataset_sha256": plan["dataset_sha256"],
                "config_sha256": digest(config),
                "inference_signature": inference_signature(config),
                "optimization_ids": [
                    r["record_id"] for k in ["fit", "select"] for r in groups[k]
                ],
                "mode": config["mode"],
            }
            result["sha256"] = digest(result)
            write(result_file, result)
        else:
            result = evaluate_fn(
                backend, groups[args.split], task, config, selected["program"], args.out
            )
            result.update(
                split=args.split,
                winner_sha256=selected["sha256"],
                dataset_sha256=plan["dataset_sha256"],
                test_used_for_selection=False,
            )
            write(result_file, result)
            (args.out / "table.md").write_text(
                table(result, config["task"]["metric"]), encoding="utf-8"
            )
        write(
            args.out / "status.json",
            {
                "status": "complete",
                "operation": args.command,
                "finished_at": time.time(),
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "out": str(args.out),
                    "artifact": result_file.name,
                }
            )
        )
        return result
    except BaseException as error:
        write(
            args.out / "status.json",
            {
                "status": "failed",
                "operation": args.command,
                "error": repr(error),
                "resume": "Repeat the exact command with --resume; cached predictions and proposals are retained.",
            },
        )
        raise


def entrypoint():
    """Console scripts must return None, not the API result dictionary."""
    main()


if __name__ == "__main__":
    entrypoint()
