# Optional advanced verification

Normal commands already check row contracts, full-response construction, actual output counts, decoding bounds, grading, budgets and completion. They write `checks.json`; users do not need a separate audit command.

For a saved run that needs stronger replay verification:

```bash
python -m legal_prompt_optimization audit --run runs/bar-joint --offline
```

This CPU/tokenizer-only check verifies recursive source hashes, including the runtime compiler hook; config/data/frozen-prompt source hashes; selected instructions; exact native prompt hashes and source-ID seeds; complete four-response frames; grades/token caps; actual call counts; FIT-only reflection; SELECT scores; and recomputed DEV/TEST metrics. It writes `audit.json`. A successful generation `exit.json` plus `checks.json` is distinct from optional replay verification. Changing a winner, metric, prompt or source file must make the advanced audit fail.

```bash
python -m legal_prompt_optimization verify-bundle
python -m legal_prompt_optimization verify-data --manifest manifests/ledgar.json --data data/ledgar
```

The bundle command checks shipped files. Data preparation already checks per-row hashes; its standalone command is useful after moving files. The manifests preserve existing membership; this repository does not yet include the historical manifest-selection/generation pipeline. For custom normalized data, follow DATA.md rather than claiming a new historical cohort.

## XGrammar compatibility

The pinned stack currently uses xgrammar0.2.7 plus a small process-local Phi stop-ID workaround for199999/200020. It applies in the parent and spawned workers and does not edit installed packages. xgrammar0.2.8 includes [PR905, “Mask overridden stop tokens until grammar completion”](https://github.com/mlc-ai/xgrammar/pull/905), a potential upstream fix for the observed class of failure. Removing the workaround requires validation with the newer complete stack; this repository has not upgraded dependencies or claimed that compatibility test.

GEPA uses the config’s `seed`. Model initialization remains42; proposer requests use6000+call_index; actor/judge sampling seeds derive from their recorded IDs and draws. These schedules are intentionally fixed rather than hidden configurable overrides.

## Reproduction scope

All six public-source restorations were checked against per-row inference/feedback hashes. Fresh-clone CPU tests exercise the actual pinned GEPA with synthetic models, not GPU inference. Original archived scores and selected instructions have provenance and checksums, but their raw generations are not redistributed. This is a portable implementation of the recorded settings, not a promise of identical GPU outputs or an identical fresh optimized winner.

Inference roles always use frozen Phi-4-mini. The Gemma proposer sees FIT feedback only; SELECT chooses the prompt. Optimization opens only FIT/SELECT files. DEV/TEST reports do not feed into automatic selection. The explicitly documented historical512-vs1024 binary TEST-cap choice remains exploratory evidence.
