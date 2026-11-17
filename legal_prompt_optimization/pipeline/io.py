"""Stable, lossless artifacts and explicit dataset boundaries."""

import hashlib
import json
import math
from pathlib import Path


def dumps(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        .encode("utf-8", "backslashreplace")
        .decode("utf-8")
    )


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps(value) + "\n", encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset(path):
    """One JSONL file with explicit split tags. No implicit repartitioning."""
    groups = {name: [] for name in ["fit", "select", "dev", "test"]}
    seen = set()
    kinds = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r.get("id", r.get("record_id"))
        split = r.get("split")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError(f"Line {line_number}: missing or duplicate id {rid!r}")
        if split not in groups:
            raise ValueError(f"{rid}: split must be fit/select/dev/test")
        if not isinstance(r.get("question"), str) or not r["question"].strip():
            raise ValueError(f"{rid}: empty question")
        kind = r.get("task_type", "classification")
        kinds.add(kind)
        answer = r.get("answer")
        if kind == "classification":
            labels = r.get("public_labels")
            if (
                not isinstance(labels, list)
                or len(labels) < 2
                or any(not isinstance(v, str) or not v for v in labels)
                or len(set(labels)) != len(labels)
                or answer not in labels
            ):
                raise ValueError(
                    f"{rid}: classification requires distinct string public_labels and an answer in them"
                )
        elif kind in ("regression", "probability"):
            if (
                isinstance(answer, bool)
                or not isinstance(answer, (int, float))
                or not math.isfinite(answer)
            ):
                raise ValueError(f"{rid}: numeric target must be finite")
            if kind == "probability" and not 0 <= answer <= 1:
                raise ValueError(f"{rid}: probability target outside[0,1]")
        else:
            raise ValueError(f"{rid}: unsupported task_type {kind!r}")
        seen.add(rid)
        groups[split].append({**r, "id": rid, "record_id": rid, "task_type": kind})
    if len(kinds) != 1:
        raise ValueError("Dataset must be nonempty and use a single task_type")
    return groups


def public(row):
    """Whitelist only model-visible inputs, never labels/feedback/metadata."""
    r = {k: row[k] for k in ["record_id", "question", "task_type"]}
    if row["task_type"] == "classification":
        r["public_labels"] = row["public_labels"]
    if "candidates" in row:
        r["candidates"] = [
            {k: v for k, v in c.items() if k != "cache_record"}
            for c in row["candidates"]
        ]
    return r


def validate_contract(groups, contract):
    """Optional exact source-study membership/input checks, before model requests."""
    for split, count in contract["expected_counts"].items():
        if count is not None and len(groups[split]) != count:
            raise ValueError(
                f"{split}: expected {count} cases, got {len(groups[split])}"
            )
    from ..data import cohort_digest

    for split, expected in contract["split_sha256"].items():
        if expected is not None and cohort_digest(groups[split]) != expected:
            raise ValueError(
                f"{split}: ordered cohort/input hash differs from reference config; re-prepare using the matching explicit recipe"
            )
