# Legal prompt optimization

A config-driven pipeline for **dataset preparation → GEPA prompt optimization → evaluation**. Supply a JSONL dataset and a YAML config. vLLM is the default model-serving backend; the pipeline runner connects to its endpoints and needs no GPU.

Supports classification, numeric regression and probability prediction, with response-only, judge-only, sequential and joint optimization. Generation settings, including answer forcing, apply during both optimization and evaluation.

## Install

Use Python3.11 in an environment separate from model serving:

```bash
python3.11 -m venv .venv-runner
source .venv-runner/bin/activate
pip install -e '.[data,grammar]'
legal-prompts --help
```

`legal-prompts` is the only installed command. For an existing installation, reinstall with `pip install -e '.[data,grammar]'` to remove superseded console entry points.

The `data` extra provides source readers and tokenizers. The `grammar` extra supports compact structured output in per-case profiles. Neither installs a vLLM server. `pip install -e .` suffices for custom JSONL and endpoints providing tokenization, when compact grammar is not required.

Before running optimization, [install and start the pinned vLLM student and proposer environments](docs/serving.md#install), then verify real generation:

```bash
legal-prompts smoke --config configs/experiments/ledgar.yaml --out runs/smoke-ledgar
```

The smoke test uses synthetic problems and exercises four responses, the judge and reflection, plus sampled answer forcing when enabled. No dataset preparation is needed for this check.

## Prepare data

```bash
legal-prompts prepare --dataset ledgar --out data/ledgar.jsonl
# Equivalent dataset-specific script:
python scripts/datasets/prepare_ledgar.py --out data/ledgar.jsonl
```

Builders download pinned public sources, apply explicit filters and seeded sampling, and write one JSONL file plus local preparation metadata. They do not read saved membership lists. See [dataset rules](docs/datasets.md) for all eight datasets and source/cache options.

Custom datasets use unique IDs and explicit `fit`, `select`, `dev`, or `test` tags:

```json
{"id":"example-1","split":"fit","question":"Classify this provision…","task_type":"classification","public_labels":["Confidentiality","Termination"],"answer":"Confidentiality"}
```

Numeric rows use `task_type: "regression"` or `"probability"` and numeric answers. Optional `reference_feedback` is available only to FIT reflection; targets and feedback never enter actor/judge inputs.

## Configure, optimize and evaluate

[`configs/experiments/`](configs/experiments/) contains the eight experiment profiles. [`configs/pipeline/`](configs/pipeline/) contains general examples. Edit model endpoints to match your servers.

```bash
legal-prompts validate --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml
legal-prompts optimize --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml --out runs/ledgar

legal-prompts evaluate --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml \
  --winner runs/ledgar/winner.json --split dev --out runs/ledgar-dev
legal-prompts evaluate --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml \
  --winner runs/ledgar/winner.json --split test --out runs/ledgar-test
```

The same YAML controls optimization and evaluation. GEPA uses FIT for reflection and SELECT for prompt selection. DEV supports configuration assessment; freeze choices before TEST. Optimization never passes DEV/TEST examples to GEPA or the proposer.

Optimization and evaluation automatically check server health, vLLM API compatibility, model names, context capacity and tokenization before generation. The separate `legal-prompts preflight --config CONFIG.yaml` command is an optional diagnostic; it does not test generation or answer-forcing support.

`mode: sequential` optimizes a response prompt, generates fixed four-response FIT/SELECT banks, then optimizes a judge. Use `mode: joint` with `protocol: cohort_v1` to optimize four response instructions and one judge together; [`parcel_joint.yaml`](configs/pipeline/parcel_joint.yaml) is an example. Judge-only mode requires `--response PATH/TO/RESPONSE_WINNER.json`.

The two scoring protocols are explicit:

- `legal_gepa_v1`: per-case accuracy/Pareto, compact JSON grammar, generator prefix retained by the judge, and16 evaluation draws grouped into four trials of four. The LEDGAR profile uses response seed42/batch3/24 proposals, judge seed43/batch16/64 proposals, sampled1024-token actors and greedy512-token judges. Answer forcing is disabled.
- `cohort_v1`: cohort macro-F1 or MAE, one response per problem for response optimization, and four draws for evaluation. The NPD/PARCEL profiles use sampled1024-token actors/judges and sampled128-token answer forcing.

See [the configuration reference](docs/configuration.md) for all settings and budget accounting.

## Saved prompts and results

Final instructions are in `results/sequential/<dataset>/prompts.json`, under `program.response` and `program.judge`. The [catalog](results/sequential/catalog.json) links all eight configs, prompt pairs and metric tables. [RESULTS.md](RESULTS.md) explains evaluation coverage and limitations. Numerical score reproduction by this runner is not yet verified; tests validate CPU behavior and dataset reconstruction.

Run selected datasets end to end, or evaluate the supplied prompts:

```bash
python -m legal_prompt_optimization.pipeline.suite --dry-run
python -m legal_prompt_optimization.pipeline.suite --datasets ledgar parcel
python -m legal_prompt_optimization.pipeline.suite --datasets ledgar --saved-prompts
python -m legal_prompt_optimization.pipeline.suite --datasets parcel --stages dev test
```

The suite runs datasets sequentially against existing endpoints. It validates existing data and resumes matching completed outputs. Changed configurations require a new output directory. Saved-prompt imports are stored separately from newly optimized winners and never run GEPA.

## Generation, failures and resume

Set `generation.answer_forcing.enabled` to enable one sampled answer-prefill continuation after an initial output hits its token limit. The final answer is used during optimization and evaluation. The reasoning and initial/final outputs remain in local run records; answer forcing does not establish that reasoning is sound.

Aggregation uses the remaining valid draws and fails when none are valid. Classification failures count against the score. Numeric MAE is valid-only and is always reported with failure counts and coverage. Extreme finite values are not clipped. Inputs are never silently truncated.

Repeat an interrupted optimization/evaluation command with `--resume`. Config, dataset and selected prompts must match. Completed stages and exact cached outputs, including failures, are reused. An incomplete optimization stage replays from its seed using the request cache.

## Model serving

Run models with vLLM using the [serving instructions and launch helpers](docs/serving.md). An already-running compatible vLLM server can also be used:

```bash
legal-prompts serve --config configs/experiments/ledgar.yaml --role student \
  --python .venv-student/bin/python --gpu 0 --dry-run
```

Remove `--dry-run` to launch a foreground server. Credentials are read from configured environment-variable names. Secret values are not written into configs or cache identities.

## Development

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -v
python scripts/check_public_tree.py
```

Tests cover GEPA orchestration, mock endpoints, caching, answer forcing, split isolation, dataset recipes and suite commands. Model weights, dataset text, raw predictions, run logs and generated audit metadata stay outside Git. The code is unlicensed; dataset and model terms are separate.
