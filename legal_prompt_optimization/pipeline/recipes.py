"""Explicit source-to-cohort algorithms for the six per-case experiment datasets.

No selected-ID lists or run artifacts are read. Sources, stable IDs, token
filters, random algorithms and the reference sequence of splits are explicit.
See docs/datasets.md for origin scripts, exposure and the CaseHOLD correction.
"""

import csv
import json
import random
import re
import tarfile
from pathlib import Path

from .. import simple_data as sources
from ..data import SARA_SHA, SARA_URL, SOURCES
from ..formatters import (
    format_barexamqa,
    format_casehold,
    format_lar_echr,
    format_lexam,
)

REFERENCE_DATASETS = (
    "ledgar",
    "casehold",
    "lar_echr",
    "barexamqa_closed_book",
    "sara_entailment",
    "lexam",
)
PHI = ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083")
QWEN = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def shuffled(rows, seed=42):
    result = sorted(rows, key=lambda r: r["record_id"])
    random.Random(seed).shuffle(result)
    return result


def reporting_halves(pool):
    """Original cap2048/half split, followed by cap512 *within each half*."""
    selected = shuffled(pool)[:2048]
    midpoint = len(selected) // 2
    return selected[:midpoint][:512], selected[midpoint:][:512]


def protected_ledgar_ids(test):
    """Recreate the 256-case baseline and protected256 cohort, not a saved list."""
    baseline_ids = {r["record_id"] for r in shuffled(test)[:256]}
    eligible = sorted(
        (r for r in test if r["record_id"] not in baseline_ids),
        key=lambda r: r["record_id"],
    )
    return {r["record_id"] for r in random.Random(20261029).sample(eligible, 256)}


def public_row(
    name, split, rid, question, answer, labels, *, feedback=None, source_id=None
):
    row = {
        "record_id": rid,
        "source_split": split,
        "question": question.removesuffix(
            "Return only the correct option label as your final answer."
        ).rstrip(),
        "answer": answer,
        "public_labels": labels,
        "task_type": "classification",
    }
    if source_id is not None:
        row["source_record_id"] = str(source_id)
    if feedback is not None:
        row["reference_feedback"] = feedback
    return row


def read_sources(name, cache, offline):
    """Download all required upstream partitions at pinned revisions."""
    if name == "sara_entailment":
        return tarfile.open(sources._download(SARA_URL, cache, offline, SARA_SHA))
    repo, revision = SOURCES[name]

    def get(filename):
        return sources._download(
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
            cache,
            offline,
        )

    result = {}
    if name == "lexam":
        for count in (4, 8, 16, 32):
            config = f"mcq_{count}_choices"
            result[config] = sources._parquet(
                get(config + "/test-00000-of-00001.parquet")
            ).to_pylist()
        return result
    for split in ("train", "validation", "test"):
        if name == "ledgar":
            table = sources._parquet(get(f"ledgar/{split}-00000-of-00001.parquet"))
            labels = json.loads(table.schema.metadata[b"huggingface"])["info"][
                "features"
            ]["label"]["names"]
            result[split] = (table.to_pylist(), labels)
        elif name == "casehold":
            path = get(f"data/all/{'val' if split == 'validation' else split}.csv")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                next(reader)
                result[split] = [
                    dict(
                        example_id=int(r[0]),
                        citing_prompt=r[1],
                        label=r[12],
                        source_split=split,
                        **{f"holding_{i}": r[i + 2] for i in range(5)},
                    )
                    for r in reader
                ]
        else:
            filename = f"{split}.csv" if name == "lar_echr" else f"data/qa/{split}.csv"
            result[split] = [
                r
                for r in sources._csv(get(filename))
                if name != "barexamqa_closed_book"
                or r["idx"] not in {"mbe_494", "mbe_1121"}
            ]
    return result


def source_rows(name, raw, split, training=False):
    """Preserve reference formatting and split-specific record-ID conventions."""
    if name == "ledgar":
        data, labels = raw[split]
    else:
        data = raw[split]
    rows = []
    for i, r in enumerate(data):
        source_id = None
        feedback = None
        if name == "ledgar":
            answer = labels[r["label"]]
            question = (
                "Classify the main legal purpose of the following contract provision. Select exactly one of the public category names below.\n\nContract provision:\n"
                + r["text"].strip()
                + "\n\nPublic categories (all 100):\n"
                + " | ".join(labels)
            )
            rid = f"ledgar:{split}:{i:06d}"
            if training:
                feedback = (
                    "Dataset-designated category: "
                    + answer
                    + ". No authoritative rationale supplied."
                )
        else:
            f = (
                format_casehold(r)
                if name == "casehold"
                else format_lar_echr(r)
                if name == "lar_echr"
                else format_barexamqa(r, evidence=False)
            )
            question, answer = f["question"], f["answer"]
            labels = list("ABCDE" if name == "casehold" else "ABCD")
            if name == "casehold":
                rid = f"casehold:all:{split}:{r['example_id']}"
                if training:
                    source_id = r["example_id"]
            elif training:
                source_id = r["case_id"] if name == "lar_echr" else r["idx"]
                rid = f"{name}:{split}:{i}:{source_id}"
            else:
                rid = (
                    f"barexamqa:test:{r['idx']}"
                    if name == "barexamqa_closed_book" and split == "test"
                    else f"{name}:{split}:{i:06d}"
                )
            if training:
                feedback = (
                    r["gold_passage"]
                    if name == "barexamqa_closed_book"
                    else "Dataset-designated option: "
                    + answer
                    + ". No authoritative rationale supplied."
                )
        rows.append(
            public_row(
                name,
                split,
                rid,
                question,
                answer,
                labels,
                feedback=feedback,
                source_id=source_id,
            )
        )
    return rows


def sara_rows(archive, split):
    # These are upstream official partitions distributed inside the source archive.
    ids = archive.extractfile("sara/splits/" + split).read().decode().splitlines()
    rows = []
    for cid in ids:
        if cid.startswith("tax_case_"):
            continue
        case = archive.extractfile("sara/cases/" + cid + ".pl").read().decode()
        facts = "\n".join(
            x[2:]
            for x in case.split("% Text\n", 1)[1].split("% Question", 1)[0].splitlines()
            if x.startswith("% ")
        )
        q = " ".join(
            x[2:]
            for x in case.split("% Question\n", 1)[1]
            .split("% Facts", 1)[0]
            .splitlines()
            if x.startswith("% ")
        )
        statement, answer = q.rsplit(" ", 1)
        section = re.match(r"s(\d+)_", cid).group(1)
        statute = (
            archive.extractfile("sara/statutes/source/section" + section)
            .read()
            .decode()
        )
        question = (
            "Determine whether the statement is entailed under the supplied statute and facts. Return Entailment or Contradiction as your final answer.\n\nStatute:\n"
            + statute
            + "\n\nDescription:\n"
            + facts
            + "\n\nStatement: "
            + statement
        )
        feedback = (
            "Original dataset designated answer: "
            + answer
            + ". No authoritative worked rationale supplied."
            if split == "train"
            else None
        )
        rows.append(
            public_row(
                "sara_entailment",
                split,
                "sara_entailment:" + cid,
                question,
                answer,
                ["Entailment", "Contradiction"],
                feedback=feedback,
            )
        )
    return rows


def lexam_rows(raw):
    rows = []
    for config, data in raw.items():
        count = int(config.split("_")[1])
        for r in data:
            if r["language"] != "en":
                continue
            f = format_lexam(r)
            rows.append(
                public_row(
                    "lexam",
                    "test",
                    f"{config}:{r['id']}",
                    f["question"],
                    f["answer"],
                    [str(i + 1) for i in range(count)],
                    feedback="Dataset-designated option: "
                    + f["answer"]
                    + ". No authoritative rationale supplied.",
                )
            )
    return rows


def casehold_eligible(rows, tokenizers, *, add_special_tokens):
    questions = [r["question"] for r in rows]
    keep = [True] * len(rows)
    for tok in tokenizers:
        # Batch tokenization avoids Python/network calls per example. No truncation.
        lengths = []
        for start in range(0, len(questions), 128):
            enc = tok(
                questions[start : start + 128],
                add_special_tokens=add_special_tokens,
                truncation=False,
                return_length=True,
            )
            lengths.extend(enc["length"])
        keep = [ok and n <= 1800 for ok, n in zip(keep, lengths)]
    return [r for r, ok in zip(rows, keep) if ok]


def build(name, cache, offline=False):
    if name not in REFERENCE_DATASETS:
        raise ValueError("Unknown reference dataset")
    raw = read_sources(name, Path(cache), offline)
    details = {
        "recipe": "reference_v1",
        "seed": 42,
        "sampling": "stable record-ID sort, random.Random(42).shuffle; explicit source and phase rules",
        "sources": list(SOURCES[name]) if name in SOURCES else [SARA_URL, SARA_SHA],
    }
    try:
        if name == "lexam":
            pool = lexam_rows(raw)
            research = shuffled(pool)[:320]
            excluded = {r["record_id"] for r in research}
            dev, test = reporting_halves(
                [r for r in pool if r["record_id"] not in excluded]
            )
            groups = {
                "fit": research[:128],
                "select": research[128:192],
                "dev": dev,
                "test": test,
            }
            details.update(
                research_exclusion_count=len(excluded),
                evaluation_pool=len(pool) - len(excluded),
            )
        elif name == "sara_entailment":
            train = shuffled(sara_rows(raw, "train"))
            dev, test = reporting_halves(sara_rows(raw, "test"))
            groups = {
                "fit": train[:64],
                "select": train[64:96],
                "dev": dev,
                "test": test,
            }
        else:
            training = source_rows(name, raw, "train", True)
            if name != "ledgar":
                training += source_rows(name, raw, "validation", True)
            dev = source_rows(name, raw, "validation")
            test = source_rows(name, raw, "test")
            if name == "casehold":
                from transformers import AutoTokenizer

                tokenizers = [
                    AutoTokenizer.from_pretrained(
                        model, revision=revision, local_files_only=offline
                    )
                    for model, revision in [PHI, QWEN]
                ]
                training = casehold_eligible(
                    training, tokenizers, add_special_tokens=True
                )
                dev = casehold_eligible(dev, tokenizers, add_special_tokens=True)
                test = casehold_eligible(test, tokenizers, add_special_tokens=False)
                details["eligibility"] = {
                    "max_tokens_inclusive": 1800,
                    "tokenizers": [PHI, QWEN],
                    "special_tokens": {"fit_select": True, "dev": True, "test": False},
                }
            fit_size, select_size = (40, 20) if name == "lar_echr" else (128, 64)
            train = shuffled(training)
            dev = shuffled(dev)[:512]
            if name == "ledgar":
                protected = protected_ledgar_ids(test)
                test = [r for r in test if r["record_id"] not in protected]
                details["protected_holdout"] = {
                    "baseline_seed": 42,
                    "baseline_size": 256,
                    "holdout_seed": 20261029,
                    "holdout_size": 256,
                }
            _, test = reporting_halves(test)
            groups = {
                "fit": train[:fit_size],
                "select": train[fit_size : fit_size + select_size],
                "dev": dev,
                "test": test,
            }
            if name == "casehold":
                used = {r["record_id"] for k in ["fit", "select"] for r in groups[k]}
                groups["dev"] = [r for r in dev if r["record_id"] not in used]
                details["dev_overlap_removed"] = len(dev) - len(groups["dev"])
        expected = {
            "ledgar": (128, 64, 512, 512),
            "casehold": (128, 64, 510, 512),
            "lar_echr": (40, 20, 98, 150),
            "barexamqa_closed_book": (128, 64, 123, 59),
            "sara_entailment": (64, 32, 50, 50),
            "lexam": (128, 64, 512, 512),
        }[name]
        if tuple(len(groups[k]) for k in ("fit", "select", "dev", "test")) != expected:
            raise ValueError(
                f"{name}: source/eligibility drift changed reference cohort sizes"
            )
        details["counts"] = {s: len(rows) for s, rows in groups.items()}
        return groups, details
    finally:
        if hasattr(raw, "close"):
            raw.close()
