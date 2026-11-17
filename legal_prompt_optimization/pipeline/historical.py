"""Versioned per-case legal GEPA protocol behind the archived LEDGAR table.

Uses the repository's existing row-level engine, public prompt contracts and
seed function. Transport/caching comes from the portable endpoint backend.
No dependency on the msv_legal checkout or its local artifacts is required.
"""

import json
import time
from pathlib import Path

from .. import engine as row_engine
from ..core import grade
from .engine import attempt, metric
from .io import dumps, public, read, write


class EndpointAdapter(row_engine.Adapter):
    def record(self, name, value):
        with (self.out / name).open("a", encoding="utf-8") as stream:
            stream.write(dumps(value) + "\n")

    def generate(self, requests, cache=False):
        if cache:
            raise ValueError(
                "Historical sequential protocol does not use joint cache scheduling"
            )
        if self.calls + len(requests) > self.call_cap:
            raise row_engine.BudgetExceeded("Historical student-call cap")
        if time.monotonic() >= getattr(self.backend, "deadline", float("inf")):
            raise TimeoutError("Historical stage time budget exhausted")
        records = []
        for offset in range(0, len(requests), 64):
            chunk = requests[offset : offset + 64]
            if not chunk:
                continue
            judge = not chunk[0]["sampled"]
            if any((not req["sampled"]) != judge for req in chunk):
                raise ValueError("Mixed actor/judge batch")
            outputs = self.backend.generate(
                [public(req["row"]) for req in chunk],
                [req["instruction"] for req in chunk],
                [req["seed"] for req in chunk],
                judge=judge,
            )
            if len(outputs) != len(chunk):
                raise RuntimeError("Incomplete endpoint batch")
            for req, output in zip(chunk, outputs):
                if output["finish_reason"] not in ["stop", "length"]:
                    raise RuntimeError("Incomplete endpoint response")
                tokens = output.get("usage", {}).get("completion_tokens")
                if tokens is not None and tokens > req["cap"]:
                    raise RuntimeError("Output exceeded configured cap")
                self.calls += 1
                self.search_calls += self.phase == "search"
                record = dict(
                    call_id=self.calls,
                    record_id=req["row"]["record_id"],
                    source_record_id=req["row"].get("source_record_id"),
                    trial=req["trial"],
                    draw=req["draw"],
                    stage=req["stage"],
                    instruction=req["instruction"],
                    seed=req["seed"],
                    joint=False,
                    sampled=req["sampled"],
                    cap=req["cap"],
                    phase=self.phase,
                    condition=req.get("condition"),
                    text=output["text"],
                    finish_reason=output["finish_reason"],
                    output_index=output.get("output_index", 0),
                    output_tokens=tokens,
                    cache_record=output.get("cache_record"),
                    **grade(output["text"], req["row"]),
                )
                # Historical grading parses the public JSON, including length finishes.
                # Actual token IDs are not fabricated when an endpoint supplies only usage.
                self.record("responses.jsonl", record)
                records.append(record)
        return records

    def reflect(self, prompt):
        if time.monotonic() >= getattr(self.backend, "deadline", float("inf")):
            raise TimeoutError("Historical stage time budget exhausted")
        if self.reflections >= self.config["proposer_call_cap"]:
            raise row_engine.BudgetExceeded("Historical proposer-call cap")
        self.reflections += 1
        text = self.backend.reflect(prompt, call_number=self.reflections)
        self.record(
            "reflector_calls.jsonl",
            {
                "call": self.reflections,
                "prompt": prompt,
                "text": text,
                "seed": self.config["proposer_seed_offset"] + self.reflections,
            },
        )
        return text


def stage_config(config, stage):
    settings = config["gepa"][stage]
    return {
        "mode": stage,
        "actor_max_tokens": config["generation"]["response"]["max_tokens"],
        "judge_max_tokens": config["generation"]["judge"]["max_tokens"],
        "ordinary_instruction": config["prompts"]["response"],
        "seed_instruction": config["prompts"][
            "response" if stage == "response" else "judge"
        ],
        "seed": config["seed"] if settings["seed"] is None else settings["seed"],
        "reflection_minibatch_size": settings["reflection_batch_size"],
        "student_call_cap": settings["max_solver_calls"],
        "proposer_call_cap": settings["max_proposals"],
        "max_metric_calls": settings["max_metric_calls"],
        "module_selector": settings["module_selector"],
        "proposer_seed_offset": config["generation"]["proposer"]["seed_offset"],
    }


def validate_data(groups, task, config):
    if task.kind != "classification":
        raise ValueError("legal_gepa_v1 requires classification data")
    if any("reference_feedback" not in row for row in groups["fit"]):
        raise ValueError(
            "Historical FIT reflection requires reference_feedback. Re-prepare the frozen dataset with the current builder."
        )


def optimize(backend, groups, task, config, out, response_program=None):
    out = Path(out)
    program = dict(response_program or {})
    mode = config["mode"]
    for stage in ["response", "judge"] if mode == "sequential" else [mode]:
        dest = out / stage
        if (dest / "winner.json").exists():
            program[stage] = read(dest / "winner.json")["best_candidate"]["instruction"]
            continue
        if stage == "judge" and "response" not in program:
            raise ValueError("Judge requires response winner")
        cfg = stage_config(config, stage)
        bank_calls = (
            4 * sum(len(groups[s]) for s in ["fit", "select"])
            if stage == "judge"
            else 0
        )
        if cfg["student_call_cap"] < bank_calls + len(groups["select"]):
            raise ValueError(
                "Stage budget cannot cover the four-response bank plus seed SELECT evaluation"
            )
        dest.mkdir(parents=True, exist_ok=True)
        deadline = getattr(backend, "deadline", None)
        seconds = config["gepa"][stage]["max_seconds"]
        if deadline is not None and seconds is not None:
            backend.deadline = min(deadline, time.monotonic() + seconds)
        try:
            result = row_engine.optimize(
                cfg,
                groups,
                backend,
                attempt(dest),
                generator=program.get("response"),
                adapter_class=EndpointAdapter,
            )
        finally:
            if deadline is not None:
                backend.deadline = deadline
        write(dest / "winner.json", result)
        program[stage] = result["best_candidate"]["instruction"]
    return program


def evaluate(backend, rows, task, config, program, out):
    out = Path(out)
    cached = out / "historical-readout.json"
    if cached.exists():
        raw = read(cached)
    else:
        directory = attempt(out / "readout")
        cfg = stage_config(config, "judge")
        raw = row_engine.readout(
            cfg,
            {"evaluation": rows},
            backend,
            directory,
            {"instruction": program.get("judge", config["prompts"]["judge"])},
            generator=program["response"],
            baselines=True,
            adapter_class=EndpointAdapter,
        )
        raw["records_path"] = str((directory / "responses.jsonl").relative_to(out))
        write(cached, raw)
    records = [
        json.loads(line)
        for line in (out / raw["records_path"]).read_text().splitlines()
    ]
    by_id = {r["record_id"]: r for r in rows}

    def scores(selected):
        return metric(
            task,
            [
                by_id[r.get("source_record_id") or r["record_id"]]["answer"]
                for r in selected
            ],
            [r["answer"] if r["valid"] else None for r in selected],
            "accuracy",
        )

    def bank(condition):
        selected = [r for r in records if r["condition"] == condition]
        if condition == "optimized" and raw["ordinary_equals_optimized"]:
            selected = [r for r in records if r["condition"] == "ordinary"]
        if len(selected) != 16 * len(rows):
            raise ValueError(
                "Historical readout requires sixteen actual draws per problem"
            )
        lookup = {(r["record_id"], r["draw"]): r for r in selected}
        if len(lookup) != len(selected):
            raise ValueError("Repeated historical draw identity")
        aggregates = []
        for row in rows:
            for trial in range(4):
                answer = row_engine.vote(
                    [
                        lookup[row["record_id"], 4 * trial + i]["answer"]
                        for i in range(4)
                    ]
                )
                aggregates.append(
                    {
                        "record_id": row["record_id"],
                        "answer": answer,
                        "valid": answer is not None,
                    }
                )
        return {
            "pass1": scores(selected),
            "vote4": scores(aggregates),
            "draws": {
                "attempted": len(selected),
                "failed": sum(not r["valid"] for r in selected),
                "forcing_attempts": 0,
                "forcing_recoveries": 0,
            },
        }

    judged = [r for r in records if r["condition"] == "judge"]
    if len(judged) != 4 * len(rows):
        raise ValueError("Historical readout requires four judges per problem")
    result = {
        "n": len(rows),
        "baseline": bank("ordinary"),
        "optimized_response": bank("optimized"),
        "judge": {**scores(judged), "forcing_attempts": 0, "forcing_recoveries": 0},
        "judge_optimized": "judge" in program,
        "response_changed": program["response"] != config["prompts"]["response"],
        "protocol": "legal_gepa_v1",
        "pass1_definition": "Mean correctness over all sixteen sampled responses per problem",
        "aggregation_definition": "Mean correctness over four groups of four; no vote16",
        "prediction_counts_unit": "Draws for pass1, four-response groups for vote4/judge",
        "historical_gpu_score_reproduction_validated": False,
    }
    return result
