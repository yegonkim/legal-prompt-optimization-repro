"""Single-response GEPA objectives, fixed-bank judges and explicit readouts."""

import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from ..simple_experiment import (
    CohortAdapter,
    TaskSpec,
    _seeds,
)
from ..simple_experiment import (
    metrics as classification_metrics,
)
from .io import digest, dumps, public, read, write


def task_for(groups):
    rows = [r for g in groups.values() for r in g]
    kind = rows[0]["task_type"]
    labels = tuple(dict.fromkeys(x for r in rows for x in r.get("public_labels", [])))
    return TaskSpec(kind, labels)


def metric(task, gold, predictions, name="auto"):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Nonempty aligned metric inputs required")
    valid = [
        p is not None
        and (
            task.kind == "classification"
            and p in task.labels
            or task.kind != "classification"
            and isinstance(p, (int, float))
            and not isinstance(p, bool)
            and math.isfinite(p)
        )
        for p in predictions
    ]
    n = len(gold)
    count = sum(valid)
    if task.kind == "classification":
        if name not in ["auto", "macro_f1", "accuracy"]:
            raise ValueError("Classification metric must be macro_f1 or accuracy")
        result = classification_metrics(
            task, gold, [p if v else None for p, v in zip(predictions, valid)]
        )
        result["primary"] = result["accuracy" if name == "accuracy" else "macro_f1"]
    else:
        if (
            name not in ["auto", "mae", "brier"]
            or name == "brier"
            and task.kind != "probability"
        ):
            raise ValueError("Numeric metric must be MAE, or Brier for probability")
        errors = [
            (
                abs(y - p) ** 2
                if name == "brier" or name == "auto" and task.kind == "probability"
                else abs(y - p)
            )
            for y, p, v in zip(gold, predictions, valid)
            if v
        ]
        scale = max(errors, default=0)
        value = (
            (
                scale * (math.fsum(x / scale for x in errors) / count)
                if math.isfinite(scale)
                else None
            )
            if scale
            else (0.0 if count else None)
        )
        # Infinite absolute errors are possible from opposite finite extremes.
        if scale and not math.isfinite(scale):
            value = None
        label = (
            "brier"
            if name == "brier" or name == "auto" and task.kind == "probability"
            else "mae"
        )
        result = {
            label: value,
            "primary": -value if value is not None else -sys.float_info.max,
            "scope": "valid predictions only",
            "selection_eligible": value is not None,
        }
    return {
        **result,
        "n": n,
        "valid": count,
        "failed": n - count,
        "coverage": count / n,
    }


def aggregate(records, kind):
    values = [r["prediction"] for r in records if r.get("valid", False)]
    if not values:
        return None
    if kind == "classification":
        return Counter(values).most_common(1)[0][0]
    value = statistics.median(values)
    if math.isinf(value):
        # Avoid overflowing the sum of two finite middle values.
        ordered = sorted(values)
        middle = len(ordered) // 2
        value = ordered[middle - 1] / 2 + ordered[middle] / 2
    return value


def bank_metrics(bank, task, name="auto"):
    gold = [r["answer"] for r in bank]
    result = {
        "pass1": metric(
            task,
            gold,
            [
                r["candidates"][0]["prediction"]
                if r["candidates"][0]["valid"]
                else None
                for r in bank
            ],
            name,
        ),
        "vote4" if task.kind == "classification" else "median4": metric(
            task, gold, [aggregate(r["candidates"], task.kind) for r in bank], name
        ),
    }
    records = [c for r in bank for c in r["candidates"]]
    result["draws"] = {
        "attempted": len(records),
        "failed": sum(not c["valid"] for c in records),
        **forcing_counts(records),
    }
    return result


def forcing_counts(records):
    return {
        "forcing_attempts": sum(
            r.get("answer_forcing", {}).get("attempted", False) for r in records
        ),
        "forcing_recoveries": sum(
            r.get("answer_forcing", {}).get("recovered", False) for r in records
        ),
    }


class Journal:
    def __init__(self, path, ceiling=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.calls = 0
        self.ceiling = ceiling

    def log(self, event, **values):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(dumps({"event": event, **values}) + "\n")

    def generate(self, backend, rows, instructions, seeds, judge=False):
        if self.ceiling is not None and self.calls + len(rows) > self.ceiling:
            raise RuntimeError("GEPA solver-call ceiling exceeded")
        self.calls += len(rows)
        self.log(
            "request", ids=[r["record_id"] for r in rows], seeds=seeds, judge=judge
        )
        result = backend.generate(rows, instructions, seeds, judge=judge)
        self.log("response", records=result)
        return result


def attempt(path):
    path = Path(path)
    i = 0
    while (path / f"attempt-{i}").exists():
        i += 1
    return path / f"attempt-{i}"


def bank(backend, rows, prompts, task, seed, journal):
    assert len(prompts) == 4
    columns = [
        journal.generate(
            backend, [public(r) for r in rows], [p] * len(rows), _seeds(rows, seed, i)
        )
        for i, p in enumerate(prompts)
    ]
    return [{**r, "candidates": [c[i] for c in columns]} for i, r in enumerate(rows)]


def execute(backend, rows, program, task, seed, journal):
    if "response" in program:
        return journal.generate(
            backend,
            [public(r) for r in rows],
            [program["response"]] * len(rows),
            _seeds(rows, seed, 0),
        ), rows
    frames = (
        bank(
            backend, rows, [program[f"actor{i}"] for i in range(4)], task, seed, journal
        )
        if "actor0" in program
        else rows
    )
    if any(len(r.get("candidates", [])) != 4 for r in frames):
        raise ValueError("Judge requires four complete response records")
    return journal.generate(
        backend,
        [public(r) for r in frames],
        [program["judge"]] * len(frames),
        _seeds(frames, seed, 4),
        judge=True,
    ), frames


class Adapter(CohortAdapter):
    def __init__(self, *args, metric_name="auto", **kw):
        super().__init__(*args, **kw)
        self.metric_name = metric_name

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        outputs = []
        scores = []
        traces = []
        for cohort in batch:
            rows = cohort["rows"]
            result, frames = execute(
                self.backend, rows, candidate, self.task, self.seed, self.journal
            )
            score = metric(
                self.task,
                [r["answer"] for r in rows],
                [r["prediction"] if r["valid"] else None for r in result],
                self.metric_name,
            )
            self.select_metrics[digest(candidate)] = (
                score
                if cohort["split"] == "select"
                else self.select_metrics.get(digest(candidate), score)
            )
            outputs.append(result)
            scores.append(score["primary"])
            traces.append(
                {
                    "split": cohort["split"],
                    "rows": rows,
                    "frames": frames,
                    "outputs": result,
                    "metrics": score,
                }
            )
            self.journal.log(
                "evaluation",
                split=cohort["split"],
                ids=[r["record_id"] for r in rows],
                candidate=candidate,
                metrics=score,
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    def reflect(self, prompt):
        # The historical profile passes the same explicit per-stage call number.
        if (
            getattr(self.backend, "config", {})
            .get("generation", {})
            .get("proposer", {})
            .get("seed_offset")
            is not None
        ):
            if self.reflections >= self.budget:
                raise RuntimeError("Reflection budget exhausted")
            self.reflections += 1
            self.journal.log("reflection_request", prompt=prompt)
            result = self.backend.reflect(prompt, call_number=self.reflections)
            self.journal.log("reflection_response", text=result)
            return result
        return super().reflect(prompt)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # Parent implementation verifies exact FIT hashes before exposing feedback.
        examples = super().make_reflective_dataset(
            candidate, eval_batch, components_to_update
        )

        def portable(value):
            if isinstance(value, dict):
                return {k: portable(v) for k, v in value.items() if k != "cache_record"}
            if isinstance(value, list):
                return [portable(v) for v in value]
            return value

        return portable(examples)


def optimize_stage(backend, fit, select, task, seed_program, config, stage, out):
    import time

    deadline = getattr(backend, "deadline", None)
    seconds = config["gepa"][stage]["max_seconds"]
    if deadline is not None and seconds is not None:
        backend.deadline = min(deadline, time.monotonic() + seconds)
    try:
        return _optimize_stage(
            backend, fit, select, task, seed_program, config, stage, out
        )
    finally:
        if deadline is not None:
            backend.deadline = deadline


def _optimize_stage(backend, fit, select, task, seed_program, config, stage, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "winner.json").exists():
        return read(out / "winner.json")["program"]
    import gepa

    settings = config["gepa"][stage]
    budget = settings["max_proposals"]
    batch = settings["reflection_batch_size"]
    factor = 5 if stage == "joint" else 1
    if settings["max_solver_calls"] < len(select) * factor:
        raise ValueError("Solver budget cannot evaluate the seed on SELECT")
    journal = Journal(attempt(out), settings["max_solver_calls"])
    adapter = Adapter(
        backend,
        task,
        config["seed"],
        journal,
        fit,
        budget,
        metric_name=config["task"]["metric"],
    )
    result = gepa.optimize(
        seed_candidate=seed_program,
        trainset=[
            {"split": "fit", "rows": fit[i : i + batch]}
            for i in range(0, len(fit), batch)
        ],
        valset=[{"split": "select", "rows": select}],
        adapter=adapter,
        reflection_lm=adapter.reflect,
        reflection_minibatch_size=1,
        candidate_selection_strategy=settings["candidate_selection"],
        module_selector=settings["module_selector"],
        max_metric_calls=settings["max_metric_calls"],
        skip_perfect_score=False,
        use_merge=settings["use_merge"],
        stop_callbacks=lambda _: (
            adapter.reflections >= budget
            or journal.calls + factor * (len(select) + 2 * batch)
            > settings["max_solver_calls"]
        ),
        seed=config["seed"] if settings["seed"] is None else settings["seed"],
        run_dir=str(journal.path / "gepa"),
        raise_on_exception=True,
    )
    eligible = [
        i
        for i, p in enumerate(result.candidates)
        if adapter.select_metrics.get(digest(p), {}).get("selection_eligible", True)
    ]
    if not eligible:
        raise ValueError("No candidate has a defined SELECT metric")
    best = max(eligible, key=lambda i: result.val_aggregate_scores[i])
    program = result.candidates[best]
    write(
        out / "winner.json",
        {
            "program": program,
            "best_idx": best,
            "selected_on": "SELECT",
            "scores": list(result.val_aggregate_scores),
            "initial_solver_calls": journal.calls,
            "reflection_calls": adapter.reflections,
        },
    )
    return program


def cached(path, fn):
    path = Path(path)
    if path.exists():
        return read(path)
    value = fn()
    write(path, value)
    return value


def optimize(backend, groups, task, config, out, response_program=None):
    out = Path(out)
    fit = groups["fit"]
    select = groups["select"]
    if not fit or not select:
        raise ValueError("Optimization requires nonempty FIT and SELECT")
    mode = config["mode"]
    ordinary = config["prompts"]["response"]
    neutral = config["prompts"]["judge"]
    if mode == "joint":
        return optimize_stage(
            backend,
            fit,
            select,
            task,
            {**{f"actor{i}": ordinary for i in range(4)}, "judge": neutral},
            config,
            "joint",
            out / "joint",
        )
    if mode != "judge":
        response_program = optimize_stage(
            backend,
            fit,
            select,
            task,
            {"response": ordinary},
            config,
            "response",
            out / "response",
        )
    if not response_program or set(response_program) != {"response"}:
        raise ValueError("Judge mode requires a response-only winner")
    if mode == "response":
        return response_program
    banks = {}
    for name, rows in [("fit", fit), ("select", select)]:
        banks[name] = cached(
            out / f"{name}-response-bank.json",
            lambda rows=rows, name=name: bank(
                backend,
                rows,
                [response_program["response"]] * 4,
                task,
                config["seed"],
                Journal(attempt(out / (name + "-bank-log"))),
            ),
        )
    judged = optimize_stage(
        backend,
        banks["fit"],
        banks["select"],
        task,
        {"judge": neutral},
        config,
        "judge",
        out / "judge",
    )
    return {**response_program, **judged}


def evaluate(backend, rows, task, config, program, out):
    out = Path(out)
    ordinary = config["prompts"]["response"]
    prompts = [
        program.get("response", program.get(f"actor{i}", ordinary)) for i in range(4)
    ]
    baseline = cached(
        out / "baseline-bank.json",
        lambda: bank(
            backend,
            rows,
            [ordinary] * 4,
            task,
            config["seed"],
            Journal(attempt(out / "baseline-log")),
        ),
    )
    optimized = cached(
        out / "optimized-bank.json",
        lambda: bank(
            backend,
            rows,
            prompts,
            task,
            config["seed"],
            Journal(attempt(out / "optimized-log")),
        ),
    )
    judged = cached(
        out / "judge-predictions.json",
        lambda: backend.generate(
            [public(r) for r in optimized],
            program.get("judge", config["prompts"]["judge"]),
            _seeds(optimized, config["seed"], 4),
            judge=True,
        ),
    )
    result = {
        "n": len(rows),
        "baseline": bank_metrics(baseline, task, config["task"]["metric"]),
        "optimized_response": bank_metrics(optimized, task, config["task"]["metric"]),
        "judge": {
            **metric(
                task,
                [r["answer"] for r in rows],
                [r["prediction"] if r["valid"] else None for r in judged],
                config["task"]["metric"],
            ),
            **forcing_counts(judged),
        },
        "judge_optimized": "judge" in program,
        "response_changed": prompts != [ordinary] * 4,
    }
    return result
