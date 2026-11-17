# Configuration reference

Use YAML or JSON. YAML is recommended because it supports comments. Complete examples are in `configs/pipeline/`; omitted fields receive documented defaults and every run writes `resolved-config.json`. Unknown keys, unsupported choices and invalid types fail before model requests. Secrets are not interpolated into saved configs.

| Section | Settings and behavior |
|---|---|
| `version` | Schema version, currently `1`. |
| `mode` | `response`, `judge`, `sequential` or `joint`. |
| `seed` | Seed for GEPA and stable per-ID/per-draw generation seeds. |
| `task.metric` | `auto`, `macro_f1`, `accuracy`, `mae`, or `brier`. Auto chooses macro-F1/classification, MAE/regression, Brier/probability. |
| `models.student`, `models.proposer` | Model name, pinned revision, endpoint, optional local/Hugging Face tokenizer, optional API-key environment-variable name, context limit. Automatic vLLM startup checks verify API compatibility, served model names and advertised context capacity; the operator must ensure the server loaded the configured revision. |
| `generation.response`, `generation.judge` | Output cap, temperature, top-p, top-k, repetition penalty, presence penalty and frequency penalty. Student and judge use the same served model, but decoding parameters can differ. |
| `generation.proposer` | Output cap, temperature, top-p, and `thinking`. The endpoint must support the requested chat-template flag and return final text separately from reasoning. |
| `generation.schema_mode` | `guided_json` for older vLLM, `structured_outputs` for newer vLLM, `response_format` for JSON-schema-compatible endpoints, or `none` for prompt-only JSON. Parsing remains strict. Answer forcing additionally requires prefill and guided continuations. |
| `generation.input_overflow` | Only `error` is supported. There is no silent input truncation. |
| `generation.answer_forcing` | Enabled flag, continuation token cap, temperature and top-p. Triggered only by an initial `finish_reason: length`, once per draw. Continuation uses default repetition/presence/frequency penalties. |
| `prompts` | Initial response instruction and neutral judge instruction. No fixed demonstrations are inserted by the pipeline. |
| `gepa.response`, `.judge`, `.joint` | Maximum proposals, FIT reflection batch size, candidate selection (`pareto`/`current_best`), metric-call ceiling, initial solver-call ceiling, module selector (`round_robin`/`all`), and merge flag. Merge is explicitly unsupported and must be false. |
| `evaluation` | Explicitly records the supported four-draw protocol: first draw pass@1, valid-draw vote or median. Other draw counts are rejected rather than silently ignored. Vote ties go to the first occurring tied label. |
| `runtime` | Request concurrency, timeout, physical-completion limit, wall-clock limit, and offline tokenizer loading. Limits cover the current invocation; caches survive resume. No automatic transport retries. |

The server's tokenizer is used through `/tokenize` when no tokenizer is configured. For faster student tokenization, set `models.student.tokenizer` to its Hugging Face identifier or a local directory. The example dataset configs do this. Local tokenizers require the `data` extra; model weights are not loaded by the runner.

GEPA evaluates SELECT as a complete cohort, rather than averaging per-example macro-F1. Response optimization always uses single responses. Judge optimization uses frozen four-response FIT/SELECT banks. Joint optimization uses four response components (`actor0`…`actor3`) and one judge; a joint metric evaluation therefore consumes five initial model calls per problem. For `module_selector: all`, use a proposal budget divisible by five.

`max_proposals` counts calls to the reflection model, not necessarily distinct accepted prompts. `max_solver_calls` counts initial response/judge generations requested by GEPA, including cache reuse; answer-forcing continuations are additional physical completions. A stage stops before a full next FIT/SELECT iteration would exceed its ceiling. Bank generation and evaluation are outside the stage search ceiling and are covered by runtime limits.

Evaluation verifies the winner's integrity, dataset hash, inference signature and split disjointness. Changing initial prompts, model identities, decoding, forcing, seed, context limits, metric or evaluation protocol requires a new optimization run. Endpoint addresses, local tokenizer paths and execution concurrency can change without changing the inference signature. Model-generation responses are cached by their actual request payloads.

A single JSONL file includes all split tags, so validation reads the file. Optimization receives only FIT/SELECT rows and feedback. DEV/TEST are never forwarded to the optimizer or proposer. Use DEV for any human configuration comparison, then freeze your choice before running TEST. The CLI does not automatically search configurations or pick a winner from TEST scores.

## Per-case protocol

The default `protocol: cohort_v1` uses whole-cohort scoring, first-draw pass@1 and four-draw readouts. `protocol: legal_gepa_v1` selects the per-case classification protocol: individual cases are GEPA items, the reflective minibatch contains individual FIT cases, per-case accuracy drives Pareto selection, default GEPA perfect-score skipping is retained, and the judge receives the selected response prefix. It uses the exact legal output instruction, public-label compact grammar, `legal_gepa_v1|record_id|draw` SHA256 seed formula, four-response frame and per-case reflection feedback fields. Valid public JSON is graded even when the server reports a length finish, as defined by the grader. This protocol currently supports response/judge/sequential, not joint or answer forcing.

Additional explicit config fields:

- `data.dataset`: optional dataset name used when importing per-case prompts.
- `data.expected_counts`: optional FIT/SELECT/DEV/TEST sizes checked before model requests. Null values impose no size constraint. No dataset checksum is required.
- `gepa.<stage>.seed`: independent optimizer seed; null uses the top-level seed. In the per-case protocol, generation seeds remain record-ID based and independent of optimizer seed.
- `gepa.<stage>.max_seconds`: optional per-stage deadline within the overall runtime limit. Checked at requests/reflections; an in-flight request remains subject to its HTTP timeout.
- `generation.proposer.top_k`: optional explicit top-k; null leaves the endpoint default unchanged.
- `generation.proposer.seed_offset`: optional offset added to the stage's one-based reflection-call number. The per-case config uses 6000.
- `evaluation.pass1_mode`: `first_draw` for cohort_v1; `mean_draws` for legal_gepa_v1. Per-case evaluation requires16 draws grouped four at a time. Its numeric-aggregation field is unused because the protocol requires classification.

For per-case judge optimization, `max_solver_calls` includes bank generation and `max_metric_calls` applies to search only. With FIT 128/SELECT 64, the 8,411 total ceiling leaves 7,643 search calls after 768 bank draws. The general cohort protocol retains its documented separate bank accounting. Settings of the inactive joint stage do not apply to per-case sequential runs.

`legal_gepa_v1` needs the optional `grammar` extra to compile its compact JSON grammar. Backend rendering/seed/feedback contracts are versioned by the protocol; different inference transports may produce different numerical outcomes. Known TEST exposure is disclosed in RESULTS.md. A reference recipe is not proof of a successful GPU score reproduction.
