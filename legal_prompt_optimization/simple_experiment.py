"""Standalone finite experiments; no imports from the historical execution stack.

Backend calls are aligned arrays (one completion per row). Public rows contain
record_id, question, public_labels, and, for judges, four complete candidate
records. Prediction records require prediction and text; extra raw fields survive.
GEPA items are cohorts: one full SELECT batch and small FIT reflection batches.
Budgets count reflection calls, not GPU time; all physical calls are logged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from pathlib import Path
import traceback
from typing import Protocol


class Backend(Protocol):
    def generate(self, rows: list[dict], instructions: list[str], seeds: list[int],
                 judge: bool = False) -> list[dict]: ...
    def reflect(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class TaskSpec:
    kind: str = "classification"
    labels: tuple = ()

    def __post_init__(self):
        if self.kind not in {"classification", "regression", "probability"}:
            raise ValueError("Unknown task kind")
        if self.kind == "classification" and (not self.labels or len(set(self.labels)) != len(self.labels)):
            raise ValueError("Classification requires explicit unique public task labels")


@dataclass(frozen=True)
class SearchConfig:
    name: str
    response_budget: int = 8
    judge_budget: int = 8
    joint_budget: int = 10
    fit_batch_size: int = 1
    seed: int = 42
    max_metric_calls: int = 1000
    candidate_selection_strategy: str = 'current_best'
    solver_response_ceiling: int | None = None

    def __post_init__(self):
        if not self.name or Path(self.name).name != self.name or self.name in {".", "..", "baseline"}:
            raise ValueError("Config name must be a simple unique directory name")
        if any(type(v) is not int or v < 0 for v in
               (self.response_budget, self.judge_budget, self.joint_budget)):
            raise ValueError("Reflection budgets must be nonnegative integers")
        if self.fit_batch_size < 1 or self.max_metric_calls < 1:
            raise ValueError("Positive batch size and metric budget required")
        if self.candidate_selection_strategy not in ('current_best','pareto'):
            raise ValueError('Unsupported candidate selection strategy')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8', errors='backslashreplace').decode('utf-8')


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path, value):
    Path(path).write_text(_json(value) + "\n")


def metrics(task: TaskSpec, gold: list, predictions: list) -> dict:
    """Score the whole cohort, with a fixed public label universe and zero_division=0."""
    if not gold or len(gold) != len(predictions):
        raise ValueError("Metrics require nonempty aligned cohorts")
    n = len(gold)
    if task.kind == "classification":
        if any(y not in task.labels for y in gold):
            raise ValueError("Gold outside public task labels")
        f1 = []
        for label in task.labels:
            tp = sum(y == label and p == label for y, p in zip(gold, predictions))
            fp = sum(y != label and p == label for y, p in zip(gold, predictions))
            fn = sum(y == label and p != label for y, p in zip(gold, predictions))
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        score = sum(f1) / len(f1)
        return dict(primary=score, macro_f1=score,
                    accuracy=sum(y == p for y, p in zip(gold, predictions)) / n)
    # Malformed numerical predictions fail the run rather than disappearing from
    # the denominator or receiving an arbitrary, scale-dependent MAE penalty.
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in gold + predictions):
        raise ValueError("Numeric targets/predictions must be finite numbers")
    if task.kind == "regression":
        mae = sum(abs(y - p) for y, p in zip(gold, predictions)) / n
        return dict(primary=-mae, mae=mae)
    if any(y not in (0, 1) for y in gold) or any(not 0 <= p <= 1 for p in predictions):
        raise ValueError("Probability requires binary outcomes and probabilities in [0,1]")
    brier = sum((y - p) ** 2 for y, p in zip(gold, predictions)) / n
    return dict(primary=1 - brier, brier=brier)


def _public(row, task):
    result = {k: row[k] for k in ("record_id", "question")}
    result["task_type"] = task.kind
    if task.kind == "classification":
        result["public_labels"] = row.get("public_labels", list(task.labels))
    if "candidates" in row:
        result["candidates"] = row["candidates"]
    return result


class Journal:
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.calls = 0

    def log(self, event, **values):
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(_json(dict(event=event, **values)) + "\n")
            stream.flush()

    def generate(self, backend, rows, instructions, seeds, judge=False):
        self.log("request", rows=rows, instructions=instructions, seeds=seeds, judge=judge)
        if getattr(self, 'response_ceiling', None) is not None and self.calls + len(rows) > self.response_ceiling:
            raise RuntimeError('Frozen solver-response ceiling would be exceeded')
        self.calls += len(rows)  # requested evaluations, including failures and cache reuse
        result = backend.generate(rows, instructions, seeds, judge=judge)
        self.log("response", records=result)
        if len(result) != len(rows):
            raise ValueError("Backend returned wrong output count")
        for record in result:
            if not isinstance(record, dict) or "prediction" not in record or not isinstance(record.get("text"), str):
                raise ValueError("Backend requires prediction and full text records")
        return result

    def failure(self, error):
        self.log("failure", error=repr(error), traceback=traceback.format_exc(), calls=self.calls, generation_record=getattr(error, "generation_record", None))
        _write(self.path / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(), calls=self.calls, generation_record=getattr(error, "generation_record", None)))


def _seeds(rows, seed, draw):
    return [int(_hash([seed, row["record_id"], draw])[:8], 16) for row in rows]


def actor_bank(backend, rows, prompts, task, seed, journal):
    if len(prompts) != 4:
        raise ValueError("Exactly four actor prompts required")
    columns = [journal.generate(backend, [_public(r, task) for r in rows], [p] * len(rows),
                                _seeds(rows, seed, i)) for i, p in enumerate(prompts)]
    return [dict(row, candidates=[column[i] for column in columns]) for i, row in enumerate(rows)]


def execute(backend, rows, program, task, seed, journal):
    """One response for response search; otherwise exactly four actors + one judge."""
    if "response" in program:
        outputs = journal.generate(backend, [_public(r, task) for r in rows],
                                   [program["response"]] * len(rows), _seeds(rows, seed, 0))
        return outputs, rows
    frames = rows if "actor0" not in program else actor_bank(
        backend, rows, [program[f"actor{i}"] for i in range(4)], task, seed, journal)
    if any(len(r.get("candidates", [])) != 4 for r in frames):
        raise ValueError("Judge requires a bank of four full responses per problem")
    outputs = journal.generate(backend, [_public(r, task) for r in frames],
                               [program["judge"]] * len(rows), _seeds(rows, seed, 4), judge=True)
    return outputs, frames


class CohortAdapter:
    """GEPA scores each dataset item once, using its full cohort metric."""
    propose_new_texts = None

    def __init__(self, backend, task, seed, journal, fit, budget):
        self.backend, self.task, self.seed, self.journal = backend, task, seed, journal
        self.fit = {r["record_id"]: _hash(r) for r in fit}
        self.budget, self.reflections = budget, 0
        self.select_metrics = {}

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outputs, scores, traces = [], [], []
        for cohort in batch:
            rows = cohort["rows"]
            result, frames = execute(self.backend, rows, candidate, self.task, self.seed, self.journal)
            score = metrics(self.task, [r["answer"] for r in rows], [r["prediction"] for r in result])
            if cohort['split']=='select':
                self.select_metrics[_hash(candidate)]=score
            outputs.append(result)
            scores.append(score["primary"])
            traces.append(dict(split=cohort["split"], rows=rows, frames=frames, outputs=result, metrics=score))
            self.journal.log("evaluation", split=cohort["split"], ids=[r["record_id"] for r in rows],
                             candidate=candidate, metrics=score)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        examples = []
        for trace in eval_batch.trajectories or []:
            if trace["split"] != "fit" or any(self.fit.get(r["record_id"]) != _hash(r) for r in trace["rows"]):
                raise ValueError("Only exact FIT rows may provide reflection feedback")
            examples.append(dict(Inputs=[_public(r, self.task) for r in trace["frames"]],
                                 Outputs=trace["outputs"], Feedback=dict(
                                     answers=[r["answer"] for r in trace["rows"]], metrics=trace["metrics"])))
        self.journal.log("reflection_examples", components=components_to_update, examples=examples)
        return {component: examples for component in components_to_update}

    def reflect(self, prompt):
        if self.reflections >= self.budget:
            raise RuntimeError("Frozen reflection budget exhausted")
        self.reflections += 1
        self.journal.log("reflection_request", prompt=prompt)
        result = self.backend.reflect(prompt)
        self.journal.log("reflection_response", text=result)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Empty/non-text reflection response")
        return result


def optimize(backend, fit, select, task, seed_program, config, budget, output_dir):
    """Actual GEPA, with SELECT winner chosen by full-cohort objective."""
    journal = Journal(output_dir)
    journal.response_ceiling = config.solver_response_ceiling
    try:
        import gepa
        adapter = CohortAdapter(backend, task, config.seed, journal, fit, budget)
        train = [dict(split="fit", rows=fit[i:i + config.fit_batch_size])
                 for i in range(0, len(fit), config.fit_batch_size)]
        result = gepa.optimize(
            seed_candidate=seed_program, trainset=train, valset=[dict(split="select", rows=select)],
            adapter=adapter, reflection_lm=adapter.reflect, reflection_minibatch_size=1,
            candidate_selection_strategy=config.candidate_selection_strategy, module_selector="round_robin",
            max_metric_calls=config.max_metric_calls,
            stop_callbacks=lambda _: adapter.reflections >= budget or (
                config.solver_response_ceiling is not None and
                journal.calls + len(select) + 2 * config.fit_batch_size > config.solver_response_ceiling),
            skip_perfect_score=False, use_merge=False, seed=config.seed,
            run_dir=str(journal.path / "gepa"), raise_on_exception=True)
        best_idx,best_program=result.best_idx,result.best_candidate
        if any(not m.get('selection_eligible',True) for m in adapter.select_metrics.values()):
            eligible=[i for i,c in enumerate(result.candidates)
                      if adapter.select_metrics.get(_hash(c),{}).get('selection_eligible',True)]
            if not eligible:
                raise ValueError('No candidate has a defined SELECT metric; all predictions failed')
            best_idx=max(eligible,key=lambda i:result.val_aggregate_scores[i])
            best_program=result.candidates[best_idx]
        winner = dict(program=best_program, selected_on="SELECT", scores=list(result.val_aggregate_scores),
                      best_idx=best_idx, reflections=adapter.reflections, calls=journal.calls)
        _write(journal.path / "winner.json", winner)
        return winner["program"]
    except BaseException as error:
        journal.failure(error)
        raise


def evaluate_program(backend, rows, program, task, seed, output_dir):
    journal = Journal(output_dir)
    try:
        outputs, frames = execute(backend, rows, program, task, seed, journal)
        gold = [r["answer"] for r in rows]
        result = dict(judge=metrics(task, gold, [o["prediction"] for o in outputs]), calls=journal.calls)
        if all("candidates" in f for f in frames):
            predictions = [[c["prediction"] for c in f["candidates"]] for f in frames]
            result["actors"] = [metrics(task, gold, [p[i] for p in predictions]) for i in range(4)]
            pooled_gold = [y for y in gold for _ in range(4)]
            result["individual"] = metrics(task, pooled_gold, [v for p in predictions for v in p])
            combined = ([Counter(p).most_common(1)[0][0] for p in predictions] if task.kind == "classification"
                        else [sum(p) / 4 for p in predictions])
            result["vote4" if task.kind == "classification" else "mean4"] = metrics(task, gold, combined)
            if task.kind in {"regression", "probability"}:
                result["median4"] = metrics(task, gold, [statistics.median(p) for p in predictions])
            result["response_baseline"] = {"pass1": result["actors"][0],
                                           "vote4" if task.kind == "classification" else "median4":
                                           result["vote4" if task.kind == "classification" else "median4"]}
        _write(journal.path / "metrics.json", result)
        return result
    except BaseException as error:
        journal.failure(error)
        raise


def _validate_splits(splits, task):
    seen = set()
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name} cohort")
        for row in rows:
            rid = row["record_id"]
            if rid in seen:
                raise ValueError("Duplicate or overlapping split IDs")
            seen.add(rid)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Nonempty question required")
        metrics(task, [r["answer"] for r in rows], [r["answer"] for r in rows])


def run_experiment(backend, splits, task, ordinary_prompt, judge_prompt, configs, output_dir):
    """Run baseline, then ordered finite configurations (naive first, grid next).

    Each configuration fits response -> frozen-bank judge, and an independent
    joint five-component GEPA program. SELECT picks prompts; DEV picks the final
    method/configuration by judge primary score. Ties retain the earliest entry.
    TEST is deliberately not accepted here. Failed configurations remain on disk.
    """
    if set(splits) != {"fit", "select", "dev"}:
        raise ValueError("Development accepts exactly FIT, SELECT and DEV; never TEST")
    _validate_splits(splits, task)
    if not configs or len({c.name for c in configs}) != len(configs):
        raise ValueError("Provide a finite, nonempty list of uniquely named configs")
    root = Journal(output_dir)
    manifest = dict(task=asdict(task), configs=[asdict(c) for c in configs],
                    cohorts={s: dict(ids=[r["record_id"] for r in rows], sha256=_hash(rows)) for s, rows in splits.items()},
                    ordinary_prompt=ordinary_prompt, judge_prompt=judge_prompt,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    _write(root.path / "plan.json", manifest)
    baseline = {**{f"actor{i}": ordinary_prompt for i in range(4)}, "judge": judge_prompt}
    entries, failures = [], []
    try:
        readout = evaluate_program(backend, splits["dev"], baseline, task, configs[0].seed, root.path / "baseline")
        entries.append(dict(name="baseline", method="baseline", program=baseline, seed=configs[0].seed, metrics=readout))
        for config in configs:
            directory = root.path / config.name
            directory.mkdir()
            try:
                response = optimize(backend, splits["fit"], splits["select"], task,
                                    {"response": ordinary_prompt}, config, config.response_budget, directory / "response")
                bank_log = Journal(directory / "bank")
                try:
                    bank = {s: actor_bank(backend, splits[s], [response["response"]] * 4, task, config.seed, bank_log)
                            for s in ("fit", "select")}
                except BaseException as error:
                    bank_log.failure(error)
                    raise
                judge = optimize(backend, bank["fit"], bank["select"], task,
                                 {"judge": judge_prompt}, config, config.judge_budget, directory / "judge")
                sequential = {**{f"actor{i}": response["response"] for i in range(4)}, **judge}
                readout = evaluate_program(backend, splits["dev"], sequential, task, config.seed, directory / "sequential_dev")
                entries.append(dict(name=config.name, method="sequential", program=sequential, seed=config.seed, metrics=readout))
            except Exception as error:
                failures.append(dict(name=config.name, method="sequential", error=repr(error)))
                root.failure(error)
            try:
                joint = optimize(backend, splits["fit"], splits["select"], task, baseline,
                                 config, config.joint_budget, directory / "joint")
                readout = evaluate_program(backend, splits["dev"], joint, task, config.seed, directory / "joint_dev")
                entries.append(dict(name=config.name, method="joint", program=joint, seed=config.seed, metrics=readout))
            except Exception as error:
                failures.append(dict(name=config.name, method="joint", error=repr(error)))
                root.failure(error)
        best = max(entries, key=lambda e: e["metrics"]["judge"]["primary"])
        comparisons = {}
        for method in ("baseline", "sequential", "joint"):
            eligible = [entry for entry in entries if entry["method"] == method]
            if eligible:
                comparisons[method] = max(eligible, key=lambda e: e["metrics"]["judge"]["primary"])
        selection = dict(**best, comparisons=comparisons, task=asdict(task), selected_on="DEV", reporting_only=False,
                         development_ids=[r["record_id"] for rows in splits.values() for r in rows],
                         plan_sha256=_hash(manifest))
        selection["selection_sha256"] = _hash(selection)
        result = dict(selection=selection, entries=entries, failures=failures,
                      status="complete" if not failures else "completed_with_failures")
        _write(root.path / "selection.json", selection)
        _write(root.path / "results.json", result)
        return result
    except BaseException as error:
        root.failure(error)
        raise


def report_test(backend, rows, task, selection, output_dir):
    """Report baseline and each method's frozen DEV winner, without TEST selection."""
    unsigned = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if selection.get("selection_sha256") != _hash(unsigned) or selection.get("selected_on") != "DEV":
        raise ValueError("A frozen DEV selection is required")
    if _json(selection["task"]) != _json(asdict(task)):
        raise ValueError("Task differs from frozen selection")
    _validate_splits({"test": rows}, task)
    if set(selection["development_ids"]) & {r["record_id"] for r in rows}:
        raise ValueError("TEST overlaps development")
    root = Journal(output_dir)
    reports, failures = {}, {}
    comparisons = selection["comparisons"]
    _write(root.path / "plan.json", dict(selection=selection, test_sha256=_hash(rows)))
    try:
        for method, entry in comparisons.items():
            try:
                reports[method] = evaluate_program(backend, rows, entry["program"], task,
                                                   entry["seed"], root.path / method)
            except Exception as error:
                failures[method] = repr(error)
                root.failure(error)
        result = dict(comparisons=reports, failures=failures, reporting_only=True,
                      calls=sum(r["calls"] for r in reports.values()),
                      selection_sha256=selection["selection_sha256"], test_sha256=_hash(rows),
                      selected_on="DEV", missing_methods=sorted(set(("baseline", "sequential", "joint")) - set(comparisons)),
                      status="complete" if not failures and len(reports) == 3 else "completed_with_failures")
        _write(root.path / "report.json", result)
        return result
    except BaseException as error:
        root.failure(error)
        raise
