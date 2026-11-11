# Legal prompt optimization

Run **response → judge optimization**, or **jointly optimize four response prompts and one judge**. All inference uses frozen `microsoft/Phi-4-mini-instruct`; GEPA's local Gemma proposer receives FIT feedback only. No training, adapters or gradient updates.

The judge receives the original problem and **all four complete responses, including their reasoning**. Sequential judges use the generator prefix during both optimization and inference. Fresh candidate banks are generated automatically: no old run files are needed.

## Setup

Use Linux, Python3.11 and a compatible NVIDIA driver/CUDA stack. Optimization runs the two models on one **48GB GPU**. Model weights and public dataset texts are downloaded separately and never committed. Authenticate with Hugging Face and accept the proposer model's access terms if required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation -e .
```

`requirements.txt` lists direct dependencies; the lock file records the tested runtime versions. This code has no dependency on the original research workspace.

## Prepare a dataset

```bash
python -m legal_prompt_optimization prepare \
  --manifest manifests/ledgar.json --data data/ledgar
```

Replace `ledgar` with `casehold`, `lar_echr`, `sara_entailment`, `barexamqa_closed_book`, or `lexam`. Preparation restores the shipped membership from pinned public sources and checks input hashes automatically. SARA uses the authors' full relevant statute section, not the truncated LegalBench version. See [DATA.md](DATA.md) for custom normalized data, source access and known cohort exposure.

## Response → judge optimization

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
  --out runs/ledgar-results --gpu 0
```

Results are in `metrics.json`. Runs perform automatic correctness/completion checks and write `checks.json`; **no separate audit step is required**. Evaluation reports individual accuracy, vote@4 and judge accuracy from the same candidates. It uses four independent trials:16 actor outputs and4 judge decisions per question. Invalid/truncated outputs remain included.

### Optional: the full baseline table

Add `--baselines` to `evaluate` to also compute ordinary-prompt pass@1/vote@4:

```bash
python -m legal_prompt_optimization evaluate \
  --config configs/ledgar/judge.json --data data/ledgar \
  --generator runs/ledgar-response/winner.json \
  --winner runs/ledgar-judge/winner.json \
  --baselines --out runs/ledgar-full-table --gpu 0
```

`table.md` contains ordinary pass@1, ordinary vote@4, optimized pass@1, optimized vote@4 and optimized+judge. The optimized columns share exactly the same16 samples; ordinary columns share a separate16-sample bank. Identical ordinary/optimized instructions reuse one bank. This is optional and adds16 actor calls per question when the instructions differ.

## Joint optimization

```bash
python -m legal_prompt_optimization optimize \
  --config configs/barexamqa_closed_book/joint_all.json \
  --data data/barexamqa_closed_book --out runs/bar-joint --gpu 0

python -m legal_prompt_optimization evaluate \
  --config configs/barexamqa_closed_book/joint_all.json \
  --data data/barexamqa_closed_book --winner runs/bar-joint/winner.json \
  --out runs/bar-joint-results --gpu 0
```

All four generators start with the same ordinary instruction, and the judge starts neutral. `joint_all.json` lets GEPA mutate all five components together; `joint_round_robin.json` mutates one at a time. Prompt changes and diversity are not forced.

## Included configurations and results

| Profile | Student-call cap | Proposer-call cap | Reflection minibatch | Seed | Optimization time cap |
|---|---:|---:|---:|---:|---:|
| Response |1500|24|3|42|90min|
| Judge |8411 including the fresh4-response bank|64|16 LEDGAR/Bar;4 others|43|4h|
| Joint round-robin |9911|88|3|42|5.5h|
| Joint all-component |9911|85|3|42|5.5h|

Actors have a1024-token cap; greedy judges have a512-token cap. LEDGAR uses FIT128/SELECT64; other datasets retain their frozen cohort sizes. The config's `runtime_seconds` controls optimization; evaluation defaults to4h and accepts `--runtime-seconds`. Unsupported config values are rejected instead of silently ignored.

Selected historical prompts are in `results/<dataset>/*_winner.json`. Use those files in the same evaluation commands to skip optimization. [RESULTS.md](RESULTS.md) contains the recorded numerical findings and provenance.

The priority is a **simple, self-contained runnable method**, not bit-for-bit replay of historical runs. This runner regenerates banks; historical studies sometimes reused banks. Archived GPU scores belong to the original audited experiments, and this portable implementation has not itself reproduced those scores on a GPU. Fresh sampling, hardware and kernels may change results.

## Optional development checks

```bash
python -m unittest discover -s tests -v
```

CPU tests cover split isolation, source-ID seeds, full reasoning/prefix handling, baseline tables, failure evidence and actual GEPA integration with synthetic models. No models or GPU are used by the tests. [Advanced verification](ADVANCED.md) is available for debugging or archival work; it is not part of the normal workflow.

Runs never overwrite or silently resume existing directories. Only physical GPU0/1 are accepted; use a free GPU. Use the same `--lock-dir` across concurrent checkouts/processes to coordinate them. Partial outputs are retained after failures. Data, run logs, caches, generations and model weights stay outside Git.

## Current research checkpoint

Later all-component results, failures and interruptions are archived in [RESULTS.md](RESULTS.md). Experiments and schedulers are paused by user request. See [MACHINE_HANDOFF.md](MACHINE_HANDOFF.md) before continuing on another machine; do not silently resume interrupted attempts.
