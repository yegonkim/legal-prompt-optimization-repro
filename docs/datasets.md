# Dataset preparation

All commands write **one JSONL file** and a neighboring `.provenance.json`. Existing outputs are not overwritten. Downloads are cached, validated, and reused on subsequent attempts. Failed or missing source inputs never become synthetic examples or silently shortened opinions.

Install `pip install -e '.[data]'`. The per-dataset scripts are equivalent to `legal-prompts prepare --dataset NAME`:

| Dataset | Script | Default split policy |
|---|---|---|
| LEDGAR | `scripts/datasets/prepare_ledgar.py` | Reconstruct shipped frozen cohorts |
| CaseHOLD | `scripts/datasets/prepare_casehold.py` | Reconstruct shipped frozen cohorts |
| LAR-ECHR | `scripts/datasets/prepare_lar_echr.py` | Reconstruct shipped frozen cohorts |
| SARA entailment | `scripts/datasets/prepare_sara_entailment.py` | Reconstruct shipped frozen cohorts |
| BarExamQA, closed book | `scripts/datasets/prepare_barexamqa_closed_book.py` | Reconstruct shipped frozen cohorts |
| LEXam | `scripts/datasets/prepare_lexam.py` | Reconstruct shipped frozen cohorts |
| ECtHR-NPD | `scripts/datasets/prepare_ecthr_npd.py` | Official chronological pools, filter before sampling |
| PARCEL | `scripts/datasets/prepare_parcel.py` | Claim-level repartition after full-input filtering |

## Older six datasets

```bash
legal-prompts prepare --dataset casehold --out data/casehold.jsonl
```

Pinned public source revisions and the packaged manifests reproduce the shipped ordered cohort IDs and formatted inputs. Every reconstructed input is checked against its manifest hash. This is an end-to-end source reconstruction, not a dependency on historical generation logs or run directories. The original studies had differing cohort sizes and prior exposure; these are historical cohorts, not claims of pristine evaluation. See the [archived data notes](legacy-data.md).

To deliberately create new random research partitions instead:

```bash
legal-prompts prepare --dataset lexam --split-policy random \
  --fit-size 128 --select-size 64 --dev-size 128 --test-size 128 \
  --seed 42 --out data/lexam-new.jsonl
```

Random preparation can mix upstream partitions and is **not** an official-split reproduction. Sizes must be explicit when a source is too small; the builder never silently shrinks them. SARA uses its supplied statute sections. LEXam deduplicates English questions across choice-count variants for fresh random preparation. BarExamQA excludes the two documented malformed source rows. `--source` supplies normalized source JSONL for these fresh builds; frozen mode does not accept replacement input files or sizes.

## NPD

```bash
legal-prompts prepare --dataset ecthr_npd --out data/npd.jsonl \
  --fit-size 128 --select-size 64 --dev-size 512 --test-size 512 \
  --max-input-tokens 4096 --workers 4
```

The builder downloads the pinned case-level CSV and all referenced official HUDOC DOCX judgments. It extracts FACTS/equivalent headings before the merits section, includes tables inside FACTS, and excludes appended award tables and later Article 41/50 sections. Existing allowlisted structured fields are included; target-derived CSV fields are not serialized into questions.

Eligibility requires nonempty FACTS with **strictly fewer than 4,096 tokens**, using the configured pinned tokenizer. Filtering happens over each full official pool before seed-42 sampling: FIT/SELECT from official train, DEV from validation, TEST from test. Missing section boundaries and length exclusions are recorded with case IDs. Download failures stop preparation rather than exclude a case silently. Brief FACTS referring to excluded appendices remain eligible, and this information limitation is recorded.

The initial full download can be substantial. Resume using the same cache directory. For existing sources, use `--source case_level.csv --judgments-dir PATH`; `--offline` requires every judgment and the tokenizer to be cached. No prior FACTS audit or run directory is needed.

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

Create JSONL records with `id`, `split`, `question`, `answer`, and `task_type`. Classification also requires a list of public labels. Split tags are `fit`, `select`, `dev`, `test`. Frozen preparation also retains supplied `reference_feedback`, used only for FIT reflection by the historical protocol. Extra metadata can be retained, but only the question, task type, public labels and (for a judge) candidate responses enter model requests. Gold answers never enter actor/judge inputs.

```bash
legal-prompts validate --data my-dataset.jsonl --config configs/pipeline/ledgar.yaml
```

Preparation provenance and historical exposure are separate concerns. Rebuilding public inputs does not prove the models have never seen them. Dataset files and source documents are excluded from Git; only pinned manifests and preparation code are distributed.

CaseHOLD’s packaged manifest removes two DEV IDs already present in FIT (`casehold:all:validation:46153` and `casehold:all:validation:45777`). Its DEV cohort therefore contains **510** cases. This correction is confined to the reusable pipeline; historical manifests and experiment results remain unchanged.

For the existing NPD/PARCEL study in this repository, rebuilding these cohorts does not erase prior optimization exposure: earlier NPD runs mixed official partitions, and the repartitioned PARCEL evaluation includes cases used by earlier runs. PARCEL also shares opinions across partitions. Report these results with those disclosures, rather than as pristine held-out evidence.
