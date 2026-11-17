"""Reproducible source-to-JSONL builders; no dependency on past run artifacts."""

import csv
import hashlib
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from .. import data as historical
from .. import simple_data as sources
from .io import dataset as validate_dataset
from .io import dumps, file_hash, read, write

DATASETS = [
    "ledgar",
    "casehold",
    "lar_echr",
    "sara_entailment",
    "barexamqa_closed_book",
    "lexam",
    "ecthr_npd",
    "parcel",
]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
START = {
    "THE FACTS",
    "AS TO THE FACTS",
    "FACTS",
    "SUBJECT-MATTER OF THE CASE",
    "SUBJECT MATTER OF THE CASE",
    "SUBJECT MATTER OF THE CASES",
    "FACTS AND PROCEDURE",
    "PROCEDURE AND FACTS",
    "THE CIRCUMSTANCES OF THE CASE",
}
END = {
    "THE LAW",
    "AS TO THE LAW",
    "LAW",
    "IN LAW",
    "THE COURT'S ASSESSMENT",
    "JOINDER OF THE APPLICATIONS",
    "JOINDER OF APPLICATIONS",
}


def norm(text):
    text = text.replace("\u200b", "").replace("’", "'")
    text = re.sub(r"ITMarkFactsComplaints(?:Start|End)", "", text)
    return re.sub(r"^[IVX]+\.\s*", "", re.sub(r"\s+", " ", text).strip().upper())


def extract_facts(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        body = ET.fromstring(z.read("word/document.xml")).find("w:body", NS)
    paragraphs = [
        "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()
        for p in body.findall("w:p", NS)
    ]
    paragraphs = [p for p in paragraphs if p]
    a = next((i for i, p in enumerate(paragraphs) if norm(p) in START), None)
    b = next(
        (
            i
            for i, p in enumerate(paragraphs)
            if a is not None
            and i > a
            and (norm(p) in END or re.match(r"^ALLEGED VIOLATION(?:S)? OF ", norm(p)))
        ),
        None,
    )
    if a is None or b is None:
        return None
    i = -1
    chunks = []
    for block in body:
        if block.tag.endswith("}p"):
            text = "".join(t.text or "" for t in block.findall(".//w:t", NS)).strip()
            if not text:
                continue
            i += 1
            if a < i < b:
                chunks.append(text)
        elif block.tag.endswith("}tbl") and a <= i < b:
            for tr in block.findall("w:tr", NS):
                chunks.append(
                    " | ".join(
                        " ".join(x.text or "" for x in tc.findall(".//w:t", NS))
                        for tc in tr.findall("w:tc", NS)
                    )
                )
    if any(
        re.match(r"^(?:[IVX]+\.\s*)?APPLICATION OF ARTICLE (?:41|50)\b", norm(t))
        for t in chunks
    ):
        raise ValueError("Compensation heading found inside FACTS")
    return "\n".join(chunks)


def download(url, path, offline=False, validator=None):
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if validator:
            validator(data)
        return data
    if offline:
        raise FileNotFoundError(f"Offline source missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (dataset preparation)",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                blob = response.read()
            if validator:
                validator(blob)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(blob)
            tmp.replace(path)
            return blob
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


class OpinionHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ["head", "script", "style", "iframe", "form"]:
            self.hidden.append(tag)
        if not self.hidden and tag in ["p", "br", "div", "tr", "para", "dateline"]:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.hidden:
            self.hidden = self.hidden[
                : len(self.hidden) - 1 - self.hidden[::-1].index(tag)
            ]
        if not self.hidden and tag in ["p", "div", "tr", "para"]:
            self.parts.append("\n")

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


def opinion_text(blob, citation):
    charset = re.search(rb'charset\s*=\s*["\x27]?([a-zA-Z0-9_-]+)', blob[:3000])
    encoding = charset.group(1).decode() if charset else "utf-8"
    parser = OpinionHTML()
    parser.feed(blob.decode(encoding))
    parser.close()
    text = "\n".join(
        t
        for t in (
            re.sub(r"\s+", " ", line).strip()
            for line in "".join(parser.parts).splitlines()
        )
        if t
    )
    if (
        citation not in text[:1500]
        or "Court of Appeals" not in text[:1500]
        or len(text) < 2000
    ):
        raise ValueError(f"Incomplete or mismatched opinion: {citation}")
    refs = set(re.findall(r"\[FN(\d+)\]", text))
    notes = set(re.findall(r"Footnote\s+(\d+)\s*:", text))
    if refs - notes:
        raise ValueError(f"Missing footnotes for {citation}: {refs - notes}")
    return text


def split_pool(rows, sizes, seed):
    ordered = sorted(rows, key=lambda r: r.get("id", r.get("record_id")))
    if len(ordered) < sum(sizes.values()):
        raise ValueError(
            f"Only {len(ordered)} eligible rows; requested {sizes}. Reduce sizes explicitly."
        )
    shuffled = random.Random(seed).sample(ordered, len(ordered))
    out = {}
    i = 0
    for key, n in sizes.items():
        out[key] = shuffled[i : i + n]
        i += n
    return out


def frozen_sources(name, manifest, cache, offline):
    """Fetch pinned raw sources into the same resumable cache as fresh builders."""
    if name == "sara_entailment":
        return tarfile.open(
            sources._download(historical.SARA_URL, cache, offline, historical.SARA_SHA)
        )
    repo, revision = historical.SOURCES[name]
    get = lambda filename: sources._download(
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/" + filename,
        cache,
        offline,
    )
    entries = [r for group in manifest["splits"].values() for r in group]
    result = {}
    if name == "lexam":
        for config in {r["metadata"]["source_config"] for r in entries}:
            result[config] = {
                str(r["id"]): r
                for r in sources._parquet(
                    get(config + "/test-00000-of-00001.parquet")
                ).to_pylist()
            }
        return result
    for split in {r["metadata"].get("source_split", "test") for r in entries}:
        if name == "ledgar":
            table = sources._parquet(get(f"ledgar/{split}-00000-of-00001.parquet"))
            labels = json.loads(table.schema.metadata[b"huggingface"])["info"][
                "features"
            ]["label"]["names"]
            result[split] = (table.to_pylist(), labels)
        elif name == "casehold":
            with get(f"data/all/{'val' if split == 'validation' else split}.csv").open(
                encoding="utf-8-sig", newline=""
            ) as f:
                reader = csv.reader(f)
                next(reader)
                rows = []
                for r in reader:
                    rows.append(
                        dict(
                            example_id=int(r[0]),
                            citing_prompt=r[1],
                            label=r[12],
                            source_split=split,
                            **{f"holding_{i}": r[i + 2] for i in range(5)},
                        )
                    )
                result[split] = {str(r["example_id"]): r for r in rows}
        else:
            filename = f"{split}.csv" if name == "lar_echr" else f"data/qa/{split}.csv"
            rows = sources._csv(get(filename))
            result[split] = (
                rows
                if name == "lar_echr"
                else {
                    r["idx"]: r for r in rows if r["idx"] not in {"mbe_494", "mbe_1121"}
                }
            )
    return result


def prepare(
    dataset,
    out,
    cache_dir=Path(".source-cache"),
    offline=False,
    split_policy=None,
    source=None,
    opinions=None,
    judgments_dir=None,
    fit_size=None,
    select_size=None,
    dev_size=None,
    test_size=None,
    seed=42,
    max_input_tokens=4096,
    tokenizer="microsoft/Phi-4-mini-instruct",
    tokenizer_revision="cfbefacb99257ffa30c83adab238a50856ac3083",
    workers=4,
):
    name = sources.canonical_name(dataset)
    if name not in DATASETS:
        raise ValueError("Supported datasets: " + ", ".join(DATASETS))
    out = Path(out)
    cache_dir = Path(cache_dir)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite dataset {out}")
    if workers < 1 or max_input_tokens < 1 or seed < 0:
        raise ValueError("Invalid workers/token limit/seed")
    policy = split_policy or (
        "official"
        if name == "ecthr_npd"
        else "random"
        if name == "parcel"
        else "frozen"
    )
    sizes = {
        "fit": 128 if fit_size is None else fit_size,
        "select": 64 if select_size is None else select_size,
        "dev": 512 if dev_size is None else dev_size,
        "test": 512 if test_size is None else test_size,
    }
    if any(type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError("All split sizes must be positive integers")
    provenance = {
        "dataset": name,
        "split_policy": policy,
        "seed": seed,
        "sources": [],
        "exclusions": [],
    }
    if policy == "frozen":
        if name in ["parcel", "ecthr_npd"]:
            raise ValueError(
                "NPD/PARCEL use official/random preparation, not historical manifests"
            )
        if any(
            v is not None for v in [fit_size, select_size, dev_size, test_size, source]
        ):
            raise ValueError(
                "Frozen cohorts fix membership/sizes. Use --split-policy random for custom sizes or --source."
            )
        manifest = Path(__file__).parent / "manifests" / (name + ".json")
        spec = read(manifest)
        raw_sources = frozen_sources(name, spec, cache_dir, offline)
        try:
            groups = {
                split: [
                    historical.restore_entry(name, entry, raw_sources)
                    for entry in entries
                ]
                for split, entries in spec["splits"].items()
            }
        finally:
            if hasattr(raw_sources, "close"):
                raw_sources.close()
        provenance.update(
            manifest_sha256=file_hash(manifest),
            cohort_note=read(manifest)["selection"],
            historical_cohorts=True,
        )
        provenance["sources"] = (
            list(historical.SOURCES[name])
            if name in historical.SOURCES
            else [historical.SARA_URL, historical.SARA_SHA]
        )
    elif name == "ecthr_npd":
        if policy != "official":
            raise ValueError(
                "NPD preparation preserves official chronological partitions; random mixing is not supported"
            )
        source = (
            Path(source)
            if source
            else sources._download(sources.NPD_URL, cache_dir, offline)
        )
        with source.open(encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f))
        provenance["sources"] = [{"url": sources.NPD_URL, "sha256": file_hash(source)}]
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            tokenizer, revision=tokenizer_revision, local_files_only=offline
        )
        docs = Path(judgments_dir) if judgments_dir else cache_dir / "npd-judgments"

        def one(r):
            case = r["case_id"]
            if not re.fullmatch(r"001-\d+", case):
                raise ValueError("Unexpected HUDOC id")
            url = f"https://hudoc.echr.coe.int/app/conversion/docx/?library=ECHR&id={case}&filename=CASE.docx"

            facts = None

            def validate(blob):
                nonlocal facts
                # Extraction also validates the ZIP/XML. Reuse that parse.
                facts = extract_facts(blob)

            blob = download(url, docs / (case + ".docx"), offline, validate)
            note = {
                "id": case,
                "official_split": r["split"],
                "source_sha256": hashlib.sha256(blob).hexdigest(),
            }
            if facts is None or not facts.strip():
                return None, {**note, "excluded": "No reliable nonempty FACTS section"}
            count = len(tok.encode(facts, add_special_tokens=False))
            note["facts_tokens"] = count
            if count >= max_input_tokens:
                return None, {**note, "excluded": "FACTS token threshold"}
            row = sources.normalize_npd([r])[0]
            row["question"] += (
                "\n\nFACTS from the judgment (appendix award tables excluded):\n"
                + facts
            )
            row["source_split"] = r["split"]
            return row, note

        pools = {}
        audit = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, (row, note) in enumerate(pool.map(one, raw), 1):
                audit.append(note)
                if row:
                    pools.setdefault(row["source_split"], []).append(row)
                if completed % 500 == 0 or completed == len(raw):
                    print(
                        f"NPD: processed {completed}/{len(raw)} judgments",
                        file=sys.stderr,
                        flush=True,
                    )
        provenance["eligibility"] = audit
        provenance["exclusions"] = [r for r in audit if "excluded" in r]
        validation = "validation" if "validation" in pools else "val"

        def pick(split, n):
            pool = sorted(pools.get(split, []), key=lambda r: r["id"])
            if len(pool) < n:
                raise ValueError(
                    f"Official {split} pool has only{len(pool)} eligible rows; requested{n}"
                )
            return random.Random(seed).sample(pool, n)

        training = pick("train", sizes["fit"] + sizes["select"])
        groups = {
            "fit": training[: sizes["fit"]],
            "select": training[sizes["fit"] :],
            "dev": pick(validation, sizes["dev"]),
            "test": pick("test", sizes["test"]),
        }
        provenance.update(
            eligible_pool_sizes={k: len(v) for k, v in pools.items()},
            input_policy="Allowlisted metadata and FACTS before merits; inline factual tables included; appended award tables excluded. Appendix references retained.",
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            max_facts_tokens_exclusive=max_input_tokens,
        )
    elif name == "parcel":
        if policy != "random":
            raise ValueError(
                "PARCEL has no official splits; use explicit claim-level random partition"
            )
        source = (
            Path(source)
            if source
            else sources._download(sources.PARCEL_URL, cache_dir, offline)
        )
        raw = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        mapping = read(opinions) if opinions else {}
        citations = sorted({r["citation"] for r in raw})
        retrieval = []

        def fetch(citation):
            match = re.fullmatch(r"(\d{4}) NY Slip Op (\d{5})", citation)
            if not match:
                raise ValueError(f"Unexpected citation {citation}")
            if citation in mapping:
                text = mapping[citation]
                if (
                    not isinstance(text, str)
                    or len(text) < 2000
                    or citation not in text[:1500]
                ):
                    raise ValueError(f"Invalid supplied opinion {citation}")
                return (
                    citation,
                    text,
                    {
                        "citation": citation,
                        "source": "provided opinion mapping",
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    },
                )
            year, number = match.groups()
            url = (
                f"https://www.nycourts.gov/reporter/3dseries/{year}/{year}_{number}.htm"
            )
            path = cache_dir / "parcel-html" / f"{year}_{number}.htm"
            try:
                blob = download(url, path, offline, lambda b: opinion_text(b, citation))
            except Exception as e:
                raise RuntimeError(
                    f"Could not retrieve complete opinion {citation}. Retry with cached HTML in {path.parent}, or provide --opinions citation-to-full-text JSON. No partial dataset was written."
                ) from e
            return (
                citation,
                opinion_text(blob, citation),
                {
                    "citation": citation,
                    "url": url,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                },
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for citation, text, note in pool.map(fetch, citations):
                mapping[citation] = text
                retrieval.append(note)
        raw_rows = sources.normalize_parcel(raw, mapping)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            tokenizer, revision=tokenizer_revision, local_files_only=offline
        )
        eligible = []
        for row in raw_rows:
            count = len(tok.encode(row["question"], add_special_tokens=False))
            if count < max_input_tokens:
                eligible.append(row)
            else:
                provenance["exclusions"].append(
                    {
                        "id": row["id"],
                        "tokens": count,
                        "excluded": "Full input token threshold",
                    }
                )
        groups = split_pool(eligible, sizes, seed)
        provenance.update(
            sources=[
                {"url": sources.PARCEL_URL, "sha256": file_hash(source)},
                *retrieval,
            ],
            eligible_rows=len(eligible),
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            max_input_tokens_exclusive=max_input_tokens,
            overlap_note="Claim-disjoint random splits can share judicial opinions; not an opinion-held-out benchmark.",
        )
        opinion_sets = {
            s: {re.search(r"Citation: (.*?)\n", r["question"]).group(1) for r in group}
            for s, group in groups.items()
        }
        provenance["opinion_overlap"] = {
            a + "__" + b: len(opinion_sets[a] & opinion_sets[b])
            for i, a in enumerate(groups)
            for b in list(groups)[i + 1 :]
        }
    else:
        if policy != "random":
            raise ValueError(
                "Use frozen historical cohorts or explicitly request random research splits"
            )
        rows = sources.load_rows(name, cache_dir, offline=offline, source_path=source)
        groups = split_pool(rows, sizes, seed)
        provenance["cohort_note"] = (
            "New seed-based research partitions; official source partitions may be mixed. No historical-membership equivalence claimed."
        )
        provenance["sources"] = (
            list(historical.SOURCES[name])
            if name in historical.SOURCES
            else [historical.SARA_URL, historical.SARA_SHA]
        )
    records = []
    for split, rows in groups.items():
        for row in rows:
            rid = row.get("id", row.get("record_id"))
            record = {
                "id": rid,
                "split": split,
                "question": row["question"],
                "answer": row["answer"],
                "task_type": row.get("task_type", "classification"),
            }
            if "public_labels" in row:
                record["public_labels"] = row["public_labels"]
            for key in [
                "reference_feedback",
                "source_record_id",
                "source_split",
                "source_dataset",
                "source_revision",
            ]:
                if key in row:
                    record[key] = row[key]
            # Model-visible inputs still use an explicit whitelist.
            records.append(record)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")
    validate_dataset(tmp)
    tmp.replace(out)
    provenance.update(
        counts={k: len(v) for k, v in groups.items()},
        dataset_sha256=file_hash(out),
        prior_exposure_note="Historical manifests describe earlier research exposure. Fresh source preparation cannot establish absence of pretrained-model exposure.",
    )
    write(out.with_suffix(out.suffix + ".provenance.json"), provenance)
    return {
        "dataset": name,
        "out": str(out),
        "counts": provenance["counts"],
        "sha256": provenance["dataset_sha256"],
    }
