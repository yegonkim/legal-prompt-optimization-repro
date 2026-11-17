"""YAML/JSON configuration with strict keys, types and resolved defaults."""

import copy
import math
from pathlib import Path

import yaml

from .io import digest

MODEL = "microsoft/Phi-4-mini-instruct"
DEFAULTS = {
    "version": 1,
    "protocol": "cohort_v1",
    "mode": "sequential",
    "seed": 42,
    "task": {"metric": "auto"},
    "data": {
        "dataset": None,
        "split_sha256": {"fit": None, "select": None, "dev": None, "test": None},
        "expected_counts": {"fit": None, "select": None, "dev": None, "test": None},
    },
    "models": {
        "student": {
            "model": MODEL,
            "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
            "endpoint": "http://127.0.0.1:8000",
            "tokenizer": None,
            "api_key_env": None,
            "context_tokens": 16384,
        },
        "proposer": {
            "model": "google/gemma-4-12B-it",
            "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "endpoint": "http://127.0.0.1:8001",
            "tokenizer": None,
            "api_key_env": None,
            "context_tokens": 32768,
        },
    },
    "generation": {
        "response": {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "judge": {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "proposer": {
            "max_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.95,
            "thinking": True,
            "top_k": None,
            "seed_offset": None,
        },
        "schema_mode": "guided_json",
        "input_overflow": "error",
        "answer_forcing": {
            "enabled": True,
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.95,
        },
    },
    "prompts": {
        "response": "Solve the legal problem carefully. Give a concise explanation and the answer.",
        "judge": "Evaluate all four complete responses against the problem. Resolve disagreements using the evidence and return your best answer.",
    },
    "gepa": {
        "response": {
            "seed": None,
            "max_seconds": None,
            "max_proposals": 24,
            "reflection_batch_size": 3,
            "candidate_selection": "pareto",
            "max_metric_calls": 10000,
            "max_solver_calls": 1500,
            "module_selector": "round_robin",
            "use_merge": False,
        },
        "judge": {
            "seed": None,
            "max_seconds": None,
            "max_proposals": 16,
            "reflection_batch_size": 3,
            "candidate_selection": "pareto",
            "max_metric_calls": 10000,
            "max_solver_calls": 1500,
            "module_selector": "round_robin",
            "use_merge": False,
        },
        "joint": {
            "seed": None,
            "max_seconds": None,
            "max_proposals": 24,
            "reflection_batch_size": 3,
            "candidate_selection": "pareto",
            "max_metric_calls": 10000,
            "max_solver_calls": 10000,
            "module_selector": "round_robin",
            "use_merge": False,
        },
    },
    "evaluation": {
        "draws": 4,
        "numeric_aggregation": "median",
        "classification_aggregation": "vote",
        "pass1_draw": 0,
        "pass1_mode": "first_draw",
    },
    "runtime": {
        "concurrency": 8,
        "timeout_seconds": 1200,
        "max_physical_calls": 100000,
        "max_seconds": 172800,
        "offline": False,
    },
}


def merge(base, update, path="config"):
    if not isinstance(update, dict):
        raise ValueError(f"{path} must be an object")  # noqa: TRY004 — uniform config error API
    unknown = set(update) - set(base)
    if unknown:
        raise ValueError(f"{path}: unsupported keys {sorted(unknown)}")
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = (
            merge(base[key], value, path + "." + key)
            if isinstance(base[key], dict)
            else value
        )
    return out


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def number(value, name, lo, hi=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < lo
        or (hi is not None and value > hi)
    ):
        raise ValueError(f"Invalid {name}: {value!r}")


def validate(c):
    if type(c["version"]) is not int or c["version"] != 1:
        raise ValueError("Only config version1 supported")
    if c["mode"] not in ["response", "judge", "sequential", "joint"]:
        raise ValueError("mode must be response/judge/sequential/joint")
    if c["protocol"] not in ["cohort_v1", "legal_gepa_v1"]:
        raise ValueError("Unsupported protocol")
    integer(c["seed"], "seed", 0)
    for split, count in c["data"]["expected_counts"].items():
        if count is not None:
            integer(count, "data.expected_counts." + split, 0)
    import re

    name = c["data"]["dataset"]
    if name is not None and (
        not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name)
    ):
        raise ValueError("Invalid data.dataset")
    for split, checksum in c["data"]["split_sha256"].items():
        if checksum is not None and (
            not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        ):
            raise ValueError("Invalid data.split_sha256." + split)
    if c["task"]["metric"] not in ["auto", "macro_f1", "accuracy", "mae", "brier"]:
        raise ValueError("Unsupported task.metric")
    for role, m in c["models"].items():
        if not isinstance(m["model"], str) or not m["model"]:
            raise ValueError(f"{role}.model required")
        if not isinstance(m["endpoint"], str) or not m["endpoint"].startswith(
            ("http://", "https://")
        ):
            raise ValueError(f"{role}.endpoint must be HTTP(S)")
        integer(m["context_tokens"], role + ".context_tokens")
        for key in ["revision", "tokenizer", "api_key_env"]:
            if m[key] is not None and (not isinstance(m[key], str) or not m[key]):
                raise ValueError(f"Invalid {role}.{key}")
    for role, p in c["generation"].items():
        if role not in ["response", "judge", "proposer", "answer_forcing"]:
            continue
        integer(p["max_tokens"], role + ".max_tokens")
        number(p["temperature"], role + ".temperature", 0, 2)
        number(p["top_p"], role + ".top_p", 0, 1)
        if p["top_p"] == 0:
            raise ValueError("top_p must be >0")
        context = c["models"]["proposer" if role == "proposer" else "student"][
            "context_tokens"
        ]
        if p["max_tokens"] >= context:
            raise ValueError(f"{role} output cap must fit model context")
        if role in ["response", "judge"]:
            integer(p["top_k"], role + ".top_k", -1)
            if p["top_k"] == 0:
                raise ValueError("top_k must be -1 or positive")
            number(p["repetition_penalty"], role + ".repetition_penalty", 0.01)
            for k in ["presence_penalty", "frequency_penalty"]:
                number(p[k], role + "." + k, -2, 2)
    for value in [
        c["generation"]["answer_forcing"]["enabled"],
        c["generation"]["proposer"]["thinking"],
        c["runtime"]["offline"],
    ]:
        if type(value) is not bool:
            raise ValueError("Boolean settings must be true/false")
    if c["generation"]["schema_mode"] not in [
        "guided_json",
        "structured_outputs",
        "response_format",
        "none",
    ]:
        raise ValueError("Unsupported schema_mode")
    if c["generation"]["input_overflow"] != "error":
        raise ValueError(
            "Only input_overflow:error supported; inputs are never silently truncated"
        )
    if c["generation"]["proposer"]["top_k"] is not None:
        integer(c["generation"]["proposer"]["top_k"], "proposer.top_k", -1)
    if c["generation"]["proposer"]["top_k"] == 0:
        raise ValueError("proposer.top_k must be -1 or positive")
    if c["generation"]["proposer"]["seed_offset"] is not None:
        integer(c["generation"]["proposer"]["seed_offset"], "proposer.seed_offset", 0)
    for stage, g in c["gepa"].items():
        if g["seed"] is not None:
            integer(g["seed"], stage + ".seed", 0)
        if g["max_seconds"] is not None:
            integer(g["max_seconds"], stage + ".max_seconds")
        for k in [
            "max_proposals",
            "reflection_batch_size",
            "max_metric_calls",
            "max_solver_calls",
        ]:
            integer(g[k], stage + "." + k)
        if g["candidate_selection"] not in ["pareto", "current_best"]:
            raise ValueError("Unsupported GEPA candidate_selection")
        if g["module_selector"] not in ["round_robin", "all"]:
            raise ValueError("Unsupported GEPA module_selector")
        if type(g["use_merge"]) is not bool:
            raise ValueError("use_merge must be boolean")
        if g["use_merge"]:
            raise ValueError("Merge search is not implemented; use_merge must be false")
        if (
            stage == "joint"
            and g["module_selector"] == "all"
            and g["max_proposals"] % 5
        ):
            raise ValueError(
                "Joint all-component proposal budget must be divisible by5"
            )
    e = c["evaluation"]
    if c["protocol"] == "cohort_v1":
        expected = {
            "draws": 4,
            "numeric_aggregation": "median",
            "classification_aggregation": "vote",
            "pass1_draw": 0,
            "pass1_mode": "first_draw",
        }
    else:
        expected = {
            "draws": 16,
            "numeric_aggregation": "median",
            "classification_aggregation": "vote",
            "pass1_draw": 0,
            "pass1_mode": "mean_draws",
        }
        if c["mode"] == "joint":
            raise ValueError(
                "legal_gepa_v1 supports response, judge or sequential modes"
            )
        if c["task"]["metric"] != "accuracy":
            raise ValueError("legal_gepa_v1 requires per-case accuracy")
        if c["generation"]["answer_forcing"]["enabled"]:
            raise ValueError("Historical legal_gepa_v1 has no answer forcing")
        if c["generation"]["schema_mode"] not in ["guided_json", "structured_outputs"]:
            raise ValueError(
                "Historical compact grammar requires a vLLM structured-output mode"
            )
        if c["generation"]["proposer"]["seed_offset"] is None:
            raise ValueError("Historical proposer requires explicit seed_offset")
        if any(
            c["gepa"][k]["candidate_selection"] != "pareto"
            or c["gepa"][k]["module_selector"] != "round_robin"
            for k in ["response", "judge"]
        ):
            raise ValueError("Historical GEPA uses Pareto/round_robin")
    if e != expected:
        raise ValueError(f"{c['protocol']} requires evaluation settings {expected}")
    for k in ["concurrency", "timeout_seconds", "max_physical_calls", "max_seconds"]:
        integer(c["runtime"][k], "runtime." + k)
    if any(not isinstance(v, str) or not v.strip() for v in c["prompts"].values()):
        raise ValueError("Prompts must be nonempty strings")
    return c


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate(merge(DEFAULTS, raw))


def inference_signature(c):
    # Endpoint relocation and local tokenizer paths do not change the recipe.
    return digest(
        {
            "models": {
                r: {
                    k: v
                    for k, v in m.items()
                    if k in ["model", "revision", "context_tokens"]
                }
                for r, m in c["models"].items()
            },
            "protocol": c["protocol"],
            "generation": c["generation"],
            "prompts": c["prompts"],
            "evaluation": c["evaluation"],
            "seed": c["seed"],
            "metric": c["task"]["metric"],
        }
    )
