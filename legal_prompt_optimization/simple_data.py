"""Manifest-free public data preparation (new splits, not historical reproduction).

API: prepare(dataset, out, cache_dir='.source-cache', *, dev_size=512,
test_size=512, select_size=None, seed=42, offline=False, source_path=None,
opinions_path=None) -> counts. Writes train/select/dev/test.jsonl only.

Stable-ID sort then seed-42 row shuffle; DEV/TEST are capped at 512 each.
All remaining rows become TRAIN/SELECT (default SELECT is 20%, rounded up).
Small sources require explicit smaller DEV/TEST sizes; no silent truncation.
These random research splits do not reproduce upstream chronological splits.

Classification: id/question/answer/public_labels. Numeric tasks additionally
use task_type='regression' or 'probability', numeric answer, no label vocabulary.
LegalForecast probability targets are observed binary outcomes, NOT forecasts.

Primary sources checked 2026-11-13:
https://github.com/mmikaildemir/PARCEL (claims/labels, no opinion corpus)
https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD (case_level only)
https://www.legalforecastbench.org/data/ (outcomes/forecasts, no input packets)
PARCEL requires opinions_path: JSON object mapping exact citation to full text.
LegalForecast requires source_path: normalized JSONL with genuine input packets;
its public forecast exports cannot substitute for model-visible court filings.

Only LEDGAR/LEXam need optional pyarrow, imported lazily. No dataset scripts,
model loading, packages installation, or GPU access occurs here.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile
import urllib.request

from .data import SOURCES, SARA_URL, SARA_SHA
from .formatters import format_barexamqa, format_casehold, format_lar_echr, format_lexam

PARCEL_URL = ('https://raw.githubusercontent.com/mmikaildemir/PARCEL/'
              '79171a4edfa6279e2e039a43649e604d82ba94a4/dataset.jsonl')
NPD_URL = ('https://huggingface.co/datasets/YanyiPU716/ECtHR-NPD/resolve/'
           '3ef4cc916727b543b669bfcd241783ac2ab6759a/data/case_level.csv')
DATASETS = (*SOURCES, 'sara_entailment', 'parcel', 'ecthr_npd', 'legalforecastbench')
ALIASES = {'barexamqa': 'barexamqa_closed_book', 'bar': 'barexamqa_closed_book',
           'sara': 'sara_entailment', 'lar-echr': 'lar_echr', 'ecthr-npd': 'ecthr_npd',
           'legalforecast': 'legalforecastbench'}
# Explicit allowlist from the authors' public_adapter.py and DATA_DICTIONARY.md.
# Never serialize entire source rows: they contain award/target-derived fields.
NPD_INPUTS = ('country_alpha2', 'hudoc_decision_body', 'case_importance',
              'has_separate_opinion', 'represented', 'num_applicants',
              'hudoc_application_count', 'judgment_year', 'num_violations_found',
              'violation_type', 'violated_articles', 'violated_articles_count',
              'violation_duration_months', 'applicant_sex', 'applicant_age_group',
              'applicant_birth_years', 'applicant_nationality_scope',
              'gdp_per_capita_current_usd', 'gdp_constant_2015_usd')


class SourceUnavailableError(RuntimeError):
    """A primary source lacks the inputs needed to construct benchmark rows."""


def canonical_name(dataset):
    name = dataset.lower()
    name = ALIASES.get(name, name)
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}; choose from {DATASETS}')
    return name


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected nonempty text')
    return value.strip()


def validate_rows(rows):
    """Validate requested schema without coercing labels or unknown numeric targets."""
    seen = set()
    for row in rows:
        rid = _text(row['id'])
        if rid in seen:
            raise ValueError(f'Duplicate id: {rid}')
        seen.add(rid)
        _text(row['question'])
        task = row.get('task_type', 'classification')
        answer = row['answer']
        if task == 'classification':
            labels = row.get('public_labels')
            if (not isinstance(labels, list) or len(labels) < 2
                    or any(not isinstance(x, str) or not x.strip() for x in labels)
                    or len(labels) != len(set(labels)) or answer not in labels):
                raise ValueError(f'Invalid classification labels/answer: {rid}')
        elif task in ('regression', 'probability'):
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
                raise ValueError(f'Expected finite numeric answer: {rid}')
            if task == 'probability' and not 0 <= answer <= 1:
                raise ValueError(f'Probability target outside [0,1]: {rid}')
            if row.get('public_labels'):
                raise ValueError('Numeric tasks must not have classification labels')
        else:
            raise ValueError(f'Unknown task_type: {task}')
    if not seen:
        raise ValueError('Source contains no usable rows')


def split_rows(rows, *, dev_size=512, test_size=512, select_size=None, seed=42):
    rows = list(rows)
    validate_rows(rows)
    for name, size in [('dev_size', dev_size), ('test_size', test_size)]:
        if type(size) is not int or not 1 <= size <= 512:
            raise ValueError(f'{name} must be an integer in [1,512]')
    remaining = len(rows) - dev_size - test_size
    if remaining < 2:
        raise ValueError(f'{len(rows)} source rows cannot supply DEV={dev_size}, TEST={test_size} '
                         'and nonempty TRAIN/SELECT; set explicit smaller dev_size/test_size')
    if select_size is None:
        select_size = max(1, math.ceil(remaining * .2))
    if type(select_size) is not int or not 1 <= select_size < remaining:
        raise ValueError('select_size must leave nonempty TRAIN and SELECT')
    ordered = sorted(rows, key=lambda r: r['id'])
    random.Random(seed).shuffle(ordered)
    edge = dev_size + test_size
    return {'train': ordered[edge + select_size:], 'select': ordered[edge:edge + select_size],
            'dev': ordered[:dev_size], 'test': ordered[dev_size:edge]}


def _download(url, cache_dir, offline=False, sha256=None):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:20] + '-' + url.rsplit('/', 1)[-1]
    path = cache / name
    if not path.exists():
        if offline:
            raise SourceUnavailableError(f'Offline cache missing: {path}')
        # Stream to temporary file: failed/interrupted transfers never become cache hits.
        fd, tmp = tempfile.mkstemp(prefix='.download-', dir=cache)
        try:
            with os.fdopen(fd, 'wb') as dest, urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, dest)
            if sha256 and hashlib.sha256(Path(tmp).read_bytes()).hexdigest() != sha256:
                raise ValueError(f'Source checksum mismatch: {url}')
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Source checksum mismatch: {path}')
    return path


def _csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _classification(rid, question, answer, labels):
    return {'id': str(rid), 'question': question, 'answer': answer, 'public_labels': list(labels)}


def normalize_npd(raw):
    result = []
    for r in raw:
        amount = float(r['y_amount_eur'])  # unknown is an error, never zero
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f'Invalid NPD award: {r["case_id"]}')
        features = {key: r[key] for key in NPD_INPUTS}
        result.append({'id': 'ecthr_npd:' + _text(r['case_id']), 'task_type': 'regression',
                       'question': 'Predict the total case-level non-pecuniary damage award in nominal euros.\n'
                                   'Use these structured case facts and merits findings:\n'
                                   + json.dumps(features, ensure_ascii=False, sort_keys=True),
                       'answer': amount})
    return result


def normalize_parcel(raw, opinions):
    missing = sorted({_text(r['citation']) for r in raw} - opinions.keys())
    if missing:
        raise SourceUnavailableError(f'PARCEL lacks full opinions for {len(missing)} citations; '
                                     f'provide opinions_path JSON mapping. First: {missing[0]}')
    return [_classification('parcel:' + str(r['claim_id']),
                            'Is the claim Supported, Refuted, or Not Found in the supplied opinion?\n\n'
                            f'Citation: {_text(r["citation"])}\n\nOpinion:\n{_text(opinions[r["citation"]])}'
                            f'\n\nClaim: {_text(r["input_claim"])}',
                            r['ground_truth_label'], ['Supported', 'Refuted', 'Not Found']) for r in raw]


def _sara(path):
    result = []
    with tarfile.open(path) as archive:
        # Read members in memory; never extract untrusted archive paths.
        for name in sorted(archive.getnames()):
            match = re.fullmatch(r'sara/cases/(s(\d+)_.*)\.pl', name)
            if not match:
                continue
            case = archive.extractfile(name).read().decode('utf-8')
            qpart = case.split('% Question\n', 1)[1].split('% Facts', 1)[0]
            q = ' '.join(line[2:] for line in qpart.splitlines() if line.startswith('% '))
            statement, answer = q.rsplit(' ', 1)
            if answer not in ('Entailment', 'Contradiction'):
                continue  # Numeric SARA questions are a different task.
            facts = case.split('% Text\n', 1)[1].split('% Question', 1)[0]
            facts = '\n'.join(line[2:] for line in facts.splitlines() if line.startswith('% '))
            statute = archive.extractfile('sara/statutes/source/section' + match[2]).read().decode('utf-8')
            result.append(_classification('sara_entailment:' + match[1],
                                         f'Determine whether the statement is entailed under the statute and facts.\n\n'
                                         f'Statute:\n{statute}\n\nFacts:\n{facts}\n\nStatement: {statement}',
                                         answer, ['Entailment', 'Contradiction']))
    return result


def _parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SourceUnavailableError('Reading LEDGAR/LEXam requires pyarrow in the existing environment') from exc
    return pq.read_table(path)


def load_rows(dataset, cache_dir='.source-cache', *, offline=False, source_path=None, opinions_path=None):
    """Load primary data; source_path is raw CSV for NPD, raw JSONL for PARCEL,
    normalized JSONL for other datasets (including authorized LegalForecast inputs).
    """
    dataset = canonical_name(dataset)
    if dataset == 'ecthr_npd':
        result = normalize_npd(_csv(source_path or _download(NPD_URL, cache_dir, offline)))
    elif dataset == 'parcel':
        raw = _jsonl(source_path or _download(PARCEL_URL, cache_dir, offline))
        opinions = json.loads(Path(opinions_path).read_text()) if opinions_path else {}
        if not isinstance(opinions, dict):
            raise ValueError('opinions_path must contain a citation-to-full-text JSON object')
        result = normalize_parcel(raw, opinions)
    elif source_path is not None:
        result = _jsonl(source_path)
    elif dataset == 'legalforecastbench':
        raise SourceUnavailableError(
            'LegalForecastBench https://www.legalforecastbench.org/data/ publishes 387 outcomes/forecasts, '
            'not the outcome-blinded input packets. Supply normalized source_path JSONL with actual '
            'filings, task_type=probability and binary outcome answer; never use model probabilities as gold. '
            'For 387 rows use dev_size=64, test_size=64, select_size=52 (TRAIN=207).')
    elif dataset == 'sara_entailment':
        result = _sara(_download(SARA_URL, cache_dir, offline, SARA_SHA))
    else:
        result = _load_original(dataset, cache_dir, offline)
    validate_rows(result)
    expected = {'ecthr_npd': 'regression', 'legalforecastbench': 'probability'}.get(dataset, 'classification')
    if any(r.get('task_type', 'classification') != expected for r in result):
        raise ValueError(f'{dataset} requires task_type={expected}')
    if dataset == 'legalforecastbench' and any(r['answer'] not in (0, 1) for r in result):
        raise ValueError('LegalForecast gold must be binary observed outcomes, not forecast probabilities')
    return result


def _load_original(dataset, cache_dir, offline):
    repo, revision = SOURCES[dataset]
    base = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'
    def get(filename):
        return _download(base + filename, cache_dir, offline)
    result = []
    if dataset == 'lexam':
        seen_ids = set()
        for n in (4, 8, 16, 32):
            for r in _parquet(get(f'mcq_{n}_choices/test-00000-of-00001.parquet')).to_pylist():
                if r['language'] != 'en' or r['id'] in seen_ids:
                    continue
                seen_ids.add(r['id'])  # Keep the smallest-choice variant; never split variants across cohorts.
                f = format_lexam(r)
                choices = ast.literal_eval(r['choices']) if isinstance(r['choices'], str) else r['choices']
                result.append(_classification('lexam:' + str(r['id']), f['question'], f['answer'],
                                              [str(i + 1) for i in range(len(choices))]))
        return result
    for split in ('train', 'validation', 'test'):
        if dataset == 'ledgar':
            table = _parquet(get(f'ledgar/{split}-00000-of-00001.parquet'))
            labels = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names']
            for i, r in enumerate(table.to_pylist()):
                result.append(_classification(f'ledgar:{split}:{i}',
                                              'Classify this contract provision.\n\n' + _text(r['text'])
                                              + '\n\nPublic categories:\n' + ' | '.join(labels),
                                              labels[r['label']], labels))
        elif dataset == 'casehold':
            path = get(f'data/all/{"val" if split == "validation" else split}.csv')
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle)
                next(reader)
                for raw in reader:
                    if len(raw) != 13:
                        raise ValueError('Unexpected CaseHOLD CSV column count')
                    r = dict(example_id=raw[0], citing_prompt=raw[1], label=raw[12], source_split=split,
                             **{f'holding_{i}': raw[i + 2] for i in range(5)})
                    f = format_casehold(r)
                    result.append(_classification(f'casehold:{split}:{raw[0]}', f['question'], f['answer'], 'ABCDE'))
        else:
            filename = f'{split}.csv' if dataset == 'lar_echr' else f'data/qa/{split}.csv'
            for i, r in enumerate(_csv(get(filename))):
                if dataset == 'lar_echr':
                    f = format_lar_echr(r)
                    rid = f'lar_echr:{split}:{i}'
                else:
                    if r['idx'] in ('mbe_494', 'mbe_1121'):
                        continue  # Two documented malformed source questions.
                    f = format_barexamqa(r, evidence=False)
                    rid = f'barexamqa:{r["idx"]}'
                result.append(_classification(rid, f['question'], f['answer'], 'ABCD'))
    return result


def prepare(dataset, out, cache_dir='.source-cache', *, dev_size=512, test_size=512,
            select_size=None, seed=42, offline=False, source_path=None, opinions_path=None):
    """Prepare four JSONL files; never overwrite a destination or require a manifest."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    data = load_rows(dataset, cache_dir, offline=offline, source_path=source_path, opinions_path=opinions_path)
    splits = split_rows(data, dev_size=dev_size, test_size=test_size, select_size=select_size, seed=seed)
    out.mkdir(parents=True, exist_ok=False)
    for split, records in splits.items():
        with (out / (split + '.jsonl')).open('x', encoding='utf-8') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return {name: len(records) for name, records in splits.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', '--data', required=True)
    parser.add_argument('--cache-dir', default='.source-cache')
    parser.add_argument('--dev-size', type=int, default=512)
    parser.add_argument('--test-size', type=int, default=512)
    parser.add_argument('--select-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--source-path')
    parser.add_argument('--opinions-path')
    args = vars(parser.parse_args(argv))
    print(json.dumps(prepare(**args), sort_keys=True))


if __name__ == '__main__':
    main()
