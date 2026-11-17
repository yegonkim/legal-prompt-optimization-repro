# Legal prompt optimization

A small pipeline for **dataset preparation → GEPA prompt optimization → evaluation**. Supply a dataset JSONL file and a YAML configuration. The same model-generation procedure—including optional answer forcing—is used during optimization and evaluation.

Supports classification, numeric regression and probability prediction; response-only, judge-only, sequential response→judge, and joint optimization. Models are accessed through configured endpoints. The runner itself does not need a GPU.

## Install

Python 3.11 is the tested runner version. Create a separate environment from your model servers:

```bash
python3.11 -m venv .venv-runner
source .venv-runner/bin/activate
pip install -e '.[data,grammar]'
legal-prompts --help
```

`pip install -e .` is enough for custom JSONL datasets and endpoints with tokenization support. The `data` extra adds public-source readers and local tokenizers. It does **not** install PyTorch or vLLM as serving requirements.

## Start with a dataset

Prepare any of the eight included datasets, or bring your own file:

```bash
legal-prompts prepare --dataset ledgar --out data/ledgar.jsonl
# Equivalent per-dataset entry point:
python scripts/datasets/prepare_ledgar.py --out data/ledgar.jsonl
```

Dataset builders select cases directly from pinned public sources using explicit filters and sampling rules—no saved cohort manifests. Historical experiment configs optionally verify the finished split hashes. See [the dataset rules](docs/datasets.md) for the exact algorithms and validation.

Each line has an ID, explicit split, question, target and task type:

```json
{"id":"example-1","split":"fit","question":"Classify this provision…","task_type":"classification","public_labels":["Confidentiality","Termination"],"answer":"Confidentiality"}
```

Use `fit`, `select`, `dev`, or `test` as the split. Numeric rows use `task_type: "regression"` or `"probability"` and a numeric `answer`, without `public_labels`. IDs must be unique across the whole file. Preparation writes a neighboring `.provenance.json` with source versions, sampling policy and hashes.

[Dataset preparation](docs/datasets.md) covers all eight scripts, official versus historical splits, NPD FACTS extraction, PARCEL opinion downloads, offline caches, filtering and custom sizes.

## Configure and optimize

Use [`configs/experiments/`](configs/experiments/) for the eight published study profiles. [`configs/pipeline/`](configs/pipeline/) contains general examples, including joint optimization. Edit model endpoints and settings as needed:

```bash
legal-prompts validate --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml
legal-prompts preflight --config configs/experiments/ledgar.yaml

legal-prompts optimize \
  --data data/ledgar.jsonl \
  --config configs/experiments/ledgar.yaml \
  --out runs/ledgar
```

The default `mode: sequential` optimizes one response prompt, generates fixed banks of four responses for FIT/SELECT, and optimizes a judge over those banks. GEPA uses **FIT examples for reflection** and **SELECT for prompt selection**. DEV and TEST targets never enter the optimization or proposer feedback. The single dataset file is validated in full; its held-out rows are not passed to the optimizer.

The general `cohort_v1` examples and recent NPD/PARCEL profiles use budget-adapted defaults: Pareto selection, reflection batches of 3, up to 24 response proposals and 16 judge proposals, and 1,500 initial solver calls per stage. Response optimization scores **one response per problem**, using macro-F1 for classification or MAE for regression. Forced-answer continuations are additional physical calls and are reported separately.

Outputs include `winner.json`, the resolved config, a hashed run plan, per-stage winners, GEPA traces, fixed response banks, and a request/output cache. `winner.json` contains the prompts, inference signature, dataset hash and optimization IDs. It is the input to evaluation.

## Evaluate

```bash
legal-prompts evaluate \
  --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml \
  --winner runs/ledgar/winner.json --split dev --out runs/ledgar-dev

# After completing configuration selection, evaluate the frozen winner on TEST:
legal-prompts evaluate \
  --data data/ledgar.jsonl --config configs/experiments/ledgar.yaml \
  --winner runs/ledgar/winner.json --split test --out runs/ledgar-test
```

Evaluation saves `metrics.json`, `table.md`, response banks and judge predictions. The table contains baseline individual and aggregate@4, optimized-response individual and aggregate@4, and optimized response + judge. Classification uses vote@4; numeric tasks use median@4. In `cohort_v1`, pass@1 is draw 0 of one four-response bank. Historical profiles average individual correctness over 16 draws and aggregate/judge correctness over four groups of four. Each judge receives four complete optimized responses.

Failed draws remain recorded. Aggregation uses the remaining valid draws and fails if none are valid. Classification failures count against scores. Numeric MAE is **valid-only**, always accompanied by failure counts and coverage; extreme finite values are not clipped. A parseable answer is not necessarily correct or supported by coherent reasoning.

## Historical LEDGAR response → judge recipe

[`configs/pipeline/ledgar_historical.yaml`](configs/pipeline/ledgar_historical.yaml) describes the effective `msv_legal` Task1 configuration behind the archived **62.06% DEV / 60.21% TEST judge accuracy**. Its [provenance file](results/sequential/ledgar/historical_recipe.json) records source hashes and the correction of stale batch-size text in the original JSON. This is a supported recipe, **not an already validated reproduction of those GPU scores**.

```bash
pip install -e '.[data,grammar]'
legal-prompts prepare --dataset ledgar --out data/ledgar-historical.jsonl
legal-prompts validate --data data/ledgar-historical.jsonl --config configs/pipeline/ledgar_historical.yaml
legal-prompts optimize --data data/ledgar-historical.jsonl --config configs/pipeline/ledgar_historical.yaml --out runs/ledgar-historical
legal-prompts evaluate --data data/ledgar-historical.jsonl --config configs/pipeline/ledgar_historical.yaml --winner runs/ledgar-historical/winner.json --split dev --out runs/ledgar-historical-dev
```

This profile checks exact ordered cohort membership and input/feedback hashes. Re-prepare datasets produced before the builder retained `reference_feedback`; that field enters FIT reflection only, never actor/judge inputs. For TEST, use the same frozen winner and config with `--split test` and a separate output directory; do not choose configurations using that result.

`protocol: legal_gepa_v1` uses per-case accuracy/Pareto, native legal JSON prompts, the original compact grammar and record-ID seeds. Response optimization uses seed 42, batch 3, 24 proposer calls and a 1,500-generation ceiling. Judge optimization uses seed 43, batch 16, 64 proposer calls and an 8,411-generation ceiling **including the 768 FIT/SELECT response-bank draws**. The judge retains the generator prefix and sees four complete responses. Actors sample at temperature 0.7/top-p 0.8/top-k 20 with 1,024 tokens; judges decode greedily with 512 tokens. Answer forcing is off. Gemma reflection uses thinking, top-k 20 and seeds 6000+call number.

Historical evaluation draws **16 responses per problem**, grouped into four trials of four. Its “individual/pass@1” is mean correctness over all 16 draws; vote@4 and judge accuracy average the four trials. Failure/coverage counts therefore refer to draws or aggregation trials, with the question count stored separately. The historical judge cap was selected after a 512-vs-1024 TEST comparison; those reference scores are exploratory.

A fresh run regenerates the frozen FIT/SELECT bank under its selected response prompt and charges its full logical cost, even when exact endpoint responses are cached. The original study reused a completed bank. Endpoint execution, kernels and fresh optimization can change numerical outcomes; model/server setup remains separate from this recipe. The original servers used BF16, eager execution, and Phi EOS alignment, with student/proposer contexts of 8,192/32,768 tokens. Match those settings when investigating numerical reproduction; the optional launch helper exposes Phi alignment through `--phi-eos-alignment`.

## Answer forcing and decoding

The examples use sampling at temperature 0.7 and top-p 0.95, with no repetition penalty. Answer forcing is enabled explicitly in YAML:

```yaml
generation:
  answer_forcing:
    enabled: true
    max_tokens: 128
    temperature: 0.7
    top_p: 0.95
```

When an initial response reaches its output cap, the runner preserves its generated reasoning where recoverable, prefills the `answer` field, and requests one sampled continuation. The continuation is constrained to a public label or a JSON number. It does not extract a convenient number from malformed text, repair reasoning, or retry until an answer succeeds. Initial output, final output, forcing attempts and recoveries are retained. Set `enabled: false` to keep strict capped generation instead.

Forcing requires a vLLM-compatible endpoint supporting assistant prefill and guided choice/regex. It applies to actors and judges, including **inside GEPA**, not just to the final table. Input overflow is an error; questions and candidate responses are never silently truncated.

## Resume and other modes

Repeat an interrupted command with `--resume`. The resolved config, data and winner must match. Completed stages and exact cached outputs—including failures—are reused. An incomplete optimizer stage replays deterministically from its seed using cached responses/proposals; it does not regenerate completed samples. Runtime/physical-request limits apply per invocation; stage solver/proposal ceilings apply to the replayed optimization trajectory.

Use `mode: response`, `judge`, or `joint` for individual stages or joint optimization. Judge-only optimization also requires `--response PATH/TO/RESPONSE_WINNER.json`. [Configuration reference](docs/configuration.md) explains every setting, budget semantics and the five-component joint mode.

## Model servers

Use existing endpoints, or the optional [vLLM launch helpers](docs/serving.md). Student and proposer servers can use separate Python environments and GPUs. API credentials are read from environment-variable names configured in YAML; secret values are not written into run configs.

```bash
legal-prompts serve --config configs/experiments/ledgar.yaml --role student \
  --python /path/to/student-env/bin/python --gpu 0 --dry-run
```

Remove `--dry-run` to start a foreground server. This command does not install packages or stop existing services.

## Run the eight experiments

The [sequential suite](configs/experiments/suite.yaml) connects each dataset to its optimization/evaluation config and saved prompts. Final instructions are in `results/sequential/<dataset>/prompts.json`, under `program.response` and `program.judge`; [the catalog](results/sequential/catalog.json) links all eight. See [results and limitations](RESULTS.md) before interpreting scores.

```bash
# Review commands without downloading or generating anything:
python -m legal_prompt_optimization.pipeline.suite --dry-run

# Prepare, optimize, evaluate DEV, then read out frozen TEST:
python -m legal_prompt_optimization.pipeline.suite --datasets ledgar parcel

# Evaluate archived prompts instead of running optimization:
python -m legal_prompt_optimization.pipeline.suite --datasets ledgar --archived-prompts

# Run a subset of phases against existing prepared data/winners:
python -m legal_prompt_optimization.pipeline.suite --datasets parcel --stages dev test
```

The suite runs datasets sequentially against your configured endpoints. Configure compatible model servers before inference; the suite does not allocate GPUs or change servers. The six historical configs require the `grammar` extra (`pip install -e '.[data,grammar]'`). Dataset downloads occur only during preparation. Repeated optimization/evaluation commands reuse matching completed outputs; changed configs require a new output directory. Preparation verifies pinned sources and completed-split hashes. Frozen archived prompts are stored separately from newly optimized winners.

For joint optimization, use `configs/pipeline/parcel_joint.yaml`, or set `mode: joint` in a `cohort_v1` config and edit `gepa.joint`. This optimizes four response instructions and one judge. Historical `legal_gepa_v1` profiles intentionally support sequential/response/judge only; do not change their mode to joint. No joint results are included in the published tables.

## Development

```bash
python -m unittest discover -s tests/pipeline -v
python scripts/check_public_tree.py
```

Tests exercise GEPA with synthetic backends, mocked endpoints, caching, answer forcing, split isolation, dataset preparation and suite orchestration. They require no GPU. Compatibility modules remain for the historical protocol; host-specific schedulers and local run artifacts are excluded from Git. A clean checkout contains source, tests, configs, explicit dataset recipes, documentation, and curated sequential results/prompts.

The code is currently **unlicensed**. See [NOTICE](NOTICE); dataset and model access and license terms remain separate. No weights or dataset texts are shipped.
