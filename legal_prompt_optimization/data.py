"""Pinned sources, input fingerprints and compatibility dataset preparation.

Cohorts are selected by explicit recipes. Fingerprints validate completed data;
they are never inputs to the sampling algorithm.
"""

import json
from pathlib import Path

from .core import digest, dump, read, rows, validate_rows

SOURCES = {
    "ledgar": ("coastalcph/lex_glue", "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"),
    "casehold": ("casehold/casehold", "acc4532a67e0966e9ed7ec4ca543e8532f983b0c"),
    "lar_echr": ("AUEB-NLP/lar-echr", "3204a623238a3a589a595de58e1e78b7e11cde6b"),
    "barexamqa_closed_book": (
        "reglab/barexam_qa",
        "b7c245a7e4739f6c785faa2489e283c8daa449d1",
    ),
    "lexam": ("LEXam-Benchmark/LEXam", "31e57ae395f92ed7284a4c28d278c55da892b898"),
}
SARA_URL = "https://nlp.jhu.edu/law/sara/sara.tar.gz"
SARA_SHA = "e4f1b845016fc38b95b702bbbb27a5c9af3eadfa856ba585662b8b2cd71c187c"


def input_digest(row):
    # Used for reconstruction instead of file-byte equality (metadata/JSON whitespace may differ).
    fields = {
        k: row[k]
        for k in ["record_id", "question", "answer", "public_labels"]
        if k in row
    }
    if "source_record_id" in row:
        fields["source_record_id"] = row["source_record_id"]
    if "reference_feedback" in row:
        fields["reference_feedback"] = row["reference_feedback"]
    return digest(
        json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def cohort_digest(records):
    """One order-sensitive hash per split, with no stored case list."""
    return digest("\n".join(input_digest(row) for row in records))


def prepare(dataset, out, cache_dir, offline=False):
    """Compatibility CLI: generate split files using the same explicit recipes."""
    from .pipeline.recipes import build

    out = Path(out)
    if out.exists():
        raise FileExistsError("No data overwrite")
    groups, provenance = build(dataset, cache_dir, offline)
    out.mkdir(parents=True)
    receipt = {"dataset": dataset, "recipe": provenance, "splits": {}}
    for split, records in groups.items():
        validate_rows(records)
        (out / (split + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        receipt["splits"][split] = {
            "count": len(records),
            "sha256": cohort_digest(records),
        }
    dump(out / "dataset-receipt.json", receipt)
    return {k: len(v) for k, v in groups.items()}


def verify(receipt, directory, splits=None):
    receipt = read(receipt)
    result = {}
    for split, check in receipt["splits"].items():
        if splits is not None and split not in splits:
            continue
        records = rows(Path(directory) / (split + ".jsonl"))
        validate_rows(records)
        if len(records) != check["count"] or cohort_digest(records) != check["sha256"]:
            raise ValueError("Wrong ordered cohort/input")
        result[split] = len(records)
    return result
