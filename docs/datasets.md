# Dataset preparation

All commands write **one JSONL file** and a neighboring `.provenance.json`. Existing outputs are not overwritten. Downloads are cached, validated, and reused on subsequent attempts. Failed or missing source inputs never become synthetic examples or silently shortened opinions.

Install `pip install -e '.[data]'`. The per-dataset scripts are equivalent to `legal-prompts prepare --dataset NAME`:

| Dataset | Script | Default split policy |
|---|---|---|
| LEDGAR | `scripts/datasets/prepare_ledgar.py` | Explicit reference source/sampling recipe |
| CaseHOLD | `scripts/datasets/prepare_casehold.py` | Explicit reference source/sampling recipe |
| LAR-ECHR | `scripts/datasets/prepare_lar_echr.py` | Explicit reference source/sampling recipe |
| SARA entailment | `scripts/datasets/prepare_sara_entailment.py` | Explicit reference source/sampling recipe |
| BarExamQA, closed book | `scripts/datasets/prepare_barexamqa_closed_book.py` | Explicit reference source/sampling recipe |
| LEXam | `scripts/datasets/prepare_lexam.py` | Explicit reference source/sampling recipe |
| ECtHR-NPD | `scripts/datasets/prepare_ecthr_npd.py` | Official chronological pools, filter before sampling |
| PARCEL | `scripts/datasets/prepare_parcel.py` | Claim-level repartition after full-input filtering |

## Per-case experiment datasets

```bash
legal-prompts prepare --dataset casehold --out data/casehold.jsonl
```

The default `--split-policy reference` executes [`pipeline/recipes.py`](../legal_prompt_optimization/pipeline/recipes.py). It downloads pinned upstream sources and computes membership directly. It reads no repository case-ID lists, previous runs, or saved cohort manifests. The command fixes seed42 and reference sizes; other seeds/sizes require an explicitly new random policy.

| Dataset | FIT/SELECT construction | DEV / TEST construction |
|---|---|---|
| LEDGAR | Official train, stable-ID sort + shuffle42, first128/next64 | DEV: shuffle official validation, first512. TEST: select256 baseline IDs using shuffle42; exclude these before sampling256 protected IDs with seed20261029; exclude that protected holdout from official TEST; shuffle42/cap2048/halve; take first512 of the TEST half. |
| CaseHOLD | Pool official train+validation; require ≤1800 tokens under both pinned Phi and Qwen tokenizers; sort/shuffle42; first128/next64 | DEV: filter official validation with the same tokenizers, sort/shuffle42/first512, then remove FIT/SELECT IDs (510 remain). TEST: filter official test, sort/shuffle42/cap2048/halve, first512 of TEST half. Token counts include special tokens in FIT/SELECT/DEV, and exclude them in TEST, as configured for each phase. |
| LAR-ECHR | Pool train+validation, stable-ID sort/shuffle42, first40/next20 | DEV: shuffle all98 validation examples. TEST: shuffle300 official test examples, take the second150. |
| SARA | Read official train assignment from the authors' source archive; omit numeric `tax_case_` cases; sort/shuffle42; first64/next32 | Read official test assignments, omit numeric cases, sort/shuffle42, split100 cases into50/50. All inputs use authors' facts/question and complete relevant statute section. |
| BarExamQA | Pool train+validation; exclude malformed `mbe_494` and `mbe_1121` before assigning positional IDs; sort/shuffle42; first128/next64 | DEV: shuffle123 official validation cases. TEST: shuffle official test pool, take its second half (59). Gold passages enter FIT feedback only. |
| LEXam | All English rows from pinned4/8/16/32-choice configurations; stable-ID sort/shuffle42, first128/next64 | Exclude the first320-case reserved sample (FIT128+SELECT64+reserved128), reshuffle remaining1487 with seed42, halve743/744, take first512 of each half. Choice-count variants are preserved. |

Source revisions and selection rules are executable code, not saved case lists. CaseHOLD preparation produces DEV510; the reported results use DEV512. See RESULTS.md for this comparison limit. Four optional `data.split_sha256` values in experiment configs verify completed outputs, including order and feedback. They never enter selection or download logic.

CaseHOLD downloads **tokenizer files only** for pinned Phi and Qwen2.5-1.5B. Both tokenizer identities/revisions are explicit in `recipes.py`; no model weights or inference are needed. `--offline` requires those tokenizers and the source files to be cached. No hidden fallback to saved IDs is available.

To deliberately create new random research partitions instead:

```bash
legal-prompts prepare --dataset lexam --split-policy random \
  --fit-size 128 --select-size 64 --dev-size 128 --test-size 128 \
  --seed 42 --out data/lexam-new.jsonl
```

Random preparation can mix upstream partitions and is **not** an official-split reproduction. Sizes must be explicit when a source is too small; the builder never silently shrinks them. SARA uses its supplied statute sections. LEXam deduplicates English questions across choice-count variants for fresh random preparation. BarExamQA excludes the two documented malformed source rows. `--source` supplies normalized source JSONL for these fresh builds; reference mode does not accept replacement input files or sizes.

## NPD

```bash
legal-prompts prepare --dataset ecthr_npd --out data/npd.jsonl \
  --fit-size 128 --select-size 64 --dev-size 512 --test-size 512 \
  --max-input-tokens 4096 --workers 4
```

The builder downloads the pinned case-level CSV and all referenced official HUDOC DOCX judgments. It extracts FACTS/equivalent headings before the merits section, includes tables inside FACTS, and excludes appended award tables and later Article 41/50 sections. Existing allowlisted structured fields are included; target-derived CSV fields are not serialized into questions.

Eligibility requires nonempty FACTS with **strictly fewer than 4,096 tokens**, using the configured pinned tokenizer. Filtering happens over each full official pool before seed-42 sampling: FIT/SELECT from official train, DEV from validation, TEST from test. Missing section boundaries and length exclusions are recorded with case IDs. Download failures stop preparation rather than exclude a case silently. Brief FACTS referring to excluded appendices remain eligible, and this information limitation is recorded.

The full download can be substantial. Resume using the same cache directory. For existing sources, use `--source case_level.csv --judgments-dir PATH`; `--offline` requires every judgment and the tokenizer to be cached. No prior FACTS audit or run directory is needed.

## PARCEL

```bash
legal-prompts prepare --dataset parcel --out data/parcel.jsonl \
  --fit-size 128 --select-size 64 --dev-size 512 --test-size 512 \
  --max-input-tokens 4096 --workers 4
```

The builder downloads pinned claims and retrieves complete opinions from the official NY Courts website. It checks the opinion header/citation, substantial body text, and footnote references. The entire opinion + citation + claim must be **strictly under 4,096 tokenizer tokens**. Eligible claims are deterministically repartitioned; PARCEL has no official train/DEV/TEST split.

Claims are disjoint, but opinions can appear across splits. Opinion overlap is computed and reported. Full opinions are preserved; the builder does not truncate opinions to satisfy the filter.

Some hosts are blocked by the court website. In that case, the command fails with the missing citation and cache path. Retry from a network that can access the site, supply cached official HTML under `CACHE/parcel-html/YEAR_NUMBER.htm`, or use `--opinions opinions.json` containing a citation-to-full-opinion-text mapping. `--source` optionally supplies the raw PARCEL claim JSONL. These are source access options, not substitutes for missing opinion content.

## Bring your own dataset

Create JSONL records with `id`, `split`, `question`, `answer`, and `task_type`. Classification also requires a list of public labels. Split tags are `fit`, `select`, `dev`, `test`. Reference preparation also retains supplied `reference_feedback`, used only for FIT reflection by the reference protocol. Extra metadata can be retained, but only the question, task type, public labels and (for a judge) candidate responses enter model requests. Gold answers never enter actor/judge inputs.

```bash
legal-prompts validate --data my-dataset.jsonl --config configs/pipeline/ledgar.yaml
```

Preparation provenance and reference exposure are separate concerns. Rebuilding public inputs does not prove the models have never seen them. Dataset files and source documents are excluded from Git; only preparation code, pinned source versions, and compact verification hashes are distributed.

CaseHOLD’s recipe computes the intersection of sampled DEV and FIT/SELECT IDs and removes those overlaps; it does not hardcode the two affected IDs. Its DEV cohort contains **510** cases. The reported DEV512 results use a different denominator.

For the existing NPD/PARCEL study in this repository, rebuilding these cohorts does not erase prior optimization exposure: earlier NPD runs mixed official partitions, and the repartitioned PARCEL evaluation includes cases used by earlier runs. PARCEL also shares opinions across partitions. Report these results with those disclosures, rather than as pristine held-out evidence.
