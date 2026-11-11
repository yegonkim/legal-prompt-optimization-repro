# Legal prompt optimization

A small, self-contained implementation of the frozen **Phi-4-mini** experiments:

1. Optimize one response prompt, freeze it, then optimize a judge that receives **four complete responses, including their reasoning**.
2. Jointly optimize four response instructions and one judge, starting from neutral instructions. GEPA can mutate one component at a time (`round_robin`) or all five (`all`).

No fine-tuning, adapters, gradients or trained weights. `microsoft/Phi-4-mini-instruct` supplies **every inference role**. `google/gemma-4-12B-it` is only an optimization proposer and sees **FIT** feedback. SELECT chooses the prompt; DEV/TEST are separate reporting commands. This repository has no dependency on the original workspace or repository.

## Quick start

Use Linux, Python **3.11.17**, a compatible NVIDIA driver/CUDA stack and a **48GB GPU**. The two local models share one GPU during optimization. Have enough disk for the two base models and source data; weights are never exported or committed. Accept the proposer model's Hugging Face access conditions and authenticate through your normal Hugging Face setup.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation -e .
python -m unittest discover -s tests -v
python -m legal_prompt_optimization verify-bundle
```

`requirements.txt` lists the direct versions. `requirements-lock.txt` records the exact runtime dependency closure from the original Linux/Python environment; it does not supply system drivers. No model files or installed environments are included. CPU contract tests need only Python; the actual-GEPA integration test runs when GEPA is installed and uses synthetic models, never GPUs or downloads.

## Restore the data

Supported dataset names are `ledgar`, `casehold`, `lar_echr`, `sara_entailment`, `barexamqa_closed_book`, and `lexam`. Restore a dataset's exact ordered cohorts:

```bash
python -m legal_prompt_optimization prepare \
  --manifest manifests/ledgar.json --data data/ledgar
python -m legal_prompt_optimization verify-data \
  --manifest manifests/ledgar.json --data data/ledgar
```

The preparation command downloads **pinned public data sources**, not models. Use `--offline` with existing Hugging Face caches (and `.source-cache/sara.tar.gz` for SARA). Run with your chosen `HF_HOME` to reuse an existing cache. Restoration fails on any per-row input/feedback hash mismatch rather than substituting a different cohort. `data/`, source caches, runs and logs are ignored by Git.

Manifests contain IDs, source metadata and hashes, **not question text, gold labels, feedback passages or generations**. They directly preserve frozen membership, including the CaseHOLD length-filtered cohort; preparation therefore does not need to download the historical Qwen tokenizer or reapply eligibility filters. For your own already-normalized data, provide `fit.jsonl`, `select.jsonl`, `dev.jsonl`, `test.jsonl` using the schema in [DATA.md](DATA.md). The bundled manifests are needed to reproduce these particular cohorts.

Official validation is used for DEV where available. SARA uses the **authors' archive and complete dataset-curated relevant statute section**, not the truncated LegalBench input. Some existing cohorts have prior research exposure or FIT/SELECT overlap; see [DATA.md](DATA.md). These are not newly untouched holdouts.

## Response optimization → judge optimization

LEDGAR uses FIT128/SELECT64. The same stage budgets apply to the other datasets, with their frozen cohort sizes. The accepted LEDGAR/Bar judge uses reflection minibatch16; the context-feasible follow-up for CaseHOLD, LAR, SARA and LEXam uses minibatch4.

```bash
python -m legal_prompt_optimization optimize \
  --config configs/ledgar/response.json --data data/ledgar \
  --out runs/ledgar-response --gpu 0

python -m legal_prompt_optimization optimize \
  --config configs/ledgar/judge.json --data data/ledgar \
  --generator runs/ledgar-response/winner.json \
  --out runs/ledgar-judge --gpu 0

python -m legal_prompt_optimization evaluate \
  --config configs/ledgar/judge.json --data data/ledgar \
  --generator runs/ledgar-response/winner.json \
  --winner runs/ledgar-judge/winner.json \
  --out runs/ledgar-sequential-report --gpu 0
```

The optimized response instruction is the judge's fixed **generator prefix both during optimization and inference**. Judge optimization first generates four responses per FIT/SELECT question, freezes the bank, and charges it inside the judge's8411-call budget. The judge sees the original problem plus the exact full raw JSON responses. No reasoning stripping, summaries or answer-only projection.

Response cap: **1500 student /24 proposer /90min**, GEPA seed42, reflection minibatch3. Judge cap: **8411 bank-inclusive student /64 proposer /4h**, seed43,512-token greedy judge. Actor cap1024, sampling temperature0.7/top-p0.8/top-k20. The recorded1024-judge comparison was worse than512;512 was retained. Its historical binary choice used TEST at the user's explicit request and is marked exploratory, not an independent confirmation.

## Joint optimization

```bash
python -m legal_prompt_optimization optimize \
  --config configs/barexamqa_closed_book/joint_all.json \
  --data data/barexamqa_closed_book --out runs/bar-joint --gpu 0

python -m legal_prompt_optimization evaluate \
  --config configs/barexamqa_closed_book/joint_all.json \
  --data data/barexamqa_closed_book --winner runs/bar-joint/winner.json \
  --out runs/bar-joint-report --gpu 0
```

`joint_round_robin.json` reproduces the task2 scientific recipe:9911 student/88 proposer/5.5h. `joint_all.json` is task3's all-component recipe:9911 student/85 proposer/5.5h. Both use seed42, Pareto, minibatch3 and no merge. The85-call cap reserves complete five-proposal iterations. All four generators start with the same ordinary instruction; the judge starts neutral. No role diversification or prompt changes are forced.

The task3 recipe was demonstrated on **BarExamQA** before this repository was created. All five SELECT-selected instructions changed substantively; SELECT improved **28/64 →33/64**. That specific program had not yet been evaluated on DEV/TEST. The same all-component config is provided for all datasets, but that availability is **not a claim of completed cross-dataset results**. Later results can be added as separately versioned evidence.

## Evaluate saved prompts without optimizing again

The selected historical instructions are included:

```bash
python -m legal_prompt_optimization evaluate \
  --config configs/ledgar/judge.json --data data/ledgar \
  --generator results/ledgar/response_winner.json \
  --winner results/ledgar/judge_winner.json \
  --out runs/ledgar-saved-prompts --gpu 0

python -m legal_prompt_optimization evaluate \
  --config configs/barexamqa_closed_book/joint_all.json \
  --data data/barexamqa_closed_book \
  --winner results/barexamqa_closed_book/joint_all_winner.json \
  --out runs/bar-saved-joint --gpu 0
```

DEV/TEST use **16 actual actor outputs per question**, grouped into four independent trials of four, and four judge decisions. `metrics.json` reports pooled individual accuracy, each generator's accuracy, vote@4 and judge accuracy from the same candidates, plus invalid and length counts. Voting breaks ties by first appearance and ignores invalid answers. Truncated/invalid outputs remain included; no repair or resampling. Each readout is capped at4h. Optimizing never loads DEV/TEST rows; report outcomes do not feed into any automatic selection.

## Audit and failure behavior

Runs never overwrite or silently resume an existing directory. GPU0/1 leases and a live vacancy check are required before models load; no other job is interrupted. Every request has its seed, prompt hash, output tokens and grade recorded locally. Runtime caps, decoding failures and context overflows stop the run; partials and failures are retained. There is no silent clipping or retry loop.

```bash
python -m legal_prompt_optimization audit --run runs/bar-joint --offline
```

A clean run needs **both** `exit.json` with exit_code0 **and** `audit.json` with audit_passed. The audit checks code/config/data/winner hashes, exact native prompts, source-ID seeds, full candidate frames, output bounds, grading, SELECT selection and FIT-only reflection. Audit is CPU/tokenizer-only. The runtime includes the original Phi XGrammar compiler stop-ID correction (`199999`, `200020`) in both parent and spawned workers; it does not edit installed packages.

## What is reproducible here?

- **Verified:** all six public-source restorations match their per-row input and feedback hashes; strict framing/seeds/prefixes/split separation are CPU tested; actual GEPA is exercised with a synthetic CPU backend for all three optimization modes; archived numerical records and selected instructions have checksums.
- **Portable reruns:** the commands above implement the recorded scientific settings with native templates, frozen model revisions and pinned libraries. Fresh optimization regenerates samples and is a new run. Source data bytes contain metadata differences, so verification uses the inference/feedback fields rather than JSON-file whitespace.
- **Not claimed:** bitwise-identical GPU generations or identical selected prompts across different hardware, kernels, scheduling and fresh bank generation. The original experiments reused audited banks in some follow-ups; this minimal runner regenerates them and counts their logical cost. Archived metrics were generated by the original pinned experiment code, **not by an unperformed GPU validation of this new package**.

[RESULTS.md](RESULTS.md) maps archived results to their original provenance. [PROVENANCE.json](PROVENANCE.json) records source hashes; `results/*/source_configs.json` preserves original configurations with historical path references as evidence, not executable dependencies. The runnable portable profiles are under `configs/`.
