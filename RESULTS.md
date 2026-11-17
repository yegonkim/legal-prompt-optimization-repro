# Sequential experiment results

Reported experiment results. End-to-end GPU score reproduction with this runner is **not yet verified**; CPU behavior and data preparation are tested. Only sequential results are published. Joint optimization remains available through config.

All eight [configs](configs/experiments/suite.yaml), final prompts and metrics are indexed in the [catalog](results/sequential/catalog.json). Raw responses, checkpoints, datasets and logs are excluded.

| Dataset | Split | Metric | Baseline individual | Baseline aggregate@4 | Optimized individual | Optimized aggregate@4 | Optimized + judge |
|---|---|---|---:|---:|---:|---:|---:|
| [ledgar](results/sequential/ledgar/prompts.json) | DEV | accuracy | 50.79% | 52.29% | 58.53% | 60.74% | 62.06% |
| [ledgar](results/sequential/ledgar/prompts.json) | TEST | accuracy | 47.13% | 49.51% | 56.32% | 58.64% | 60.21% |
| [casehold](results/sequential/casehold/prompts.json) | DEV | accuracy | 60.86% | 61.72% | 61.63% | 63.53% | 63.33% |
| [casehold](results/sequential/casehold/prompts.json) | TEST | accuracy | 55.88% | 56.10% | 56.93% | 57.32% | 58.30% |
| [lar_echr](results/sequential/lar_echr/prompts.json) | DEV | accuracy | 43.81% | 43.62% | 43.81% | 43.62% | 43.37% |
| [lar_echr](results/sequential/lar_echr/prompts.json) | TEST | accuracy | 45.79% | 46.67% | 45.79% | 46.67% | 51.50% |
| [sara_entailment](results/sequential/sara_entailment/prompts.json) | DEV | accuracy | 67.00% | 70.50% | 64.38% | 66.00% | 66.00% |
| [sara_entailment](results/sequential/sara_entailment/prompts.json) | TEST | accuracy | 66.38% | 68.50% | 66.38% | 68.00% | 60.50% |
| [barexamqa_closed_book](results/sequential/barexamqa_closed_book/prompts.json) | DEV | accuracy | 49.75% | 51.63% | 49.75% | 51.63% | 47.56% |
| [barexamqa_closed_book](results/sequential/barexamqa_closed_book/prompts.json) | TEST | accuracy | 35.17% | 35.59% | 35.17% | 35.59% | 40.25% |
| [lexam](results/sequential/lexam/prompts.json) | DEV | accuracy | 21.28% | 21.88% | 25.89% | 28.37% | 26.46% |
| [lexam](results/sequential/lexam/prompts.json) | TEST | accuracy | 22.31% | 23.34% | 24.93% | 26.61% | 25.49% |
| [ecthr_npd](results/sequential/ecthr_npd/prompts.json) | DEV | mae | 6.14523e+12 | 633231 | 2.08768e+107 | 10491.8 | 12601.3 |
| [ecthr_npd](results/sequential/ecthr_npd/prompts.json) | TEST | mae | 4.53728e+14 | 1.17864e+09 | 2.23451e+202 | 8043.76 | 849967 |
| [parcel](results/sequential/parcel/prompts.json) | DEV | macro_f1 | 39.19% | 40.67% | 32.21% | 33.75% | 31.75% |
| [parcel](results/sequential/parcel/prompts.json) | TEST | macro_f1 | 35.65% | 38.76% | 30.17% | 32.40% | 28.39% |

## Interpretation and protocol differences

The six original datasets use mean individual accuracy over 16 sampled responses per question; aggregate/judge scores average four groups of four. NPD/PARCEL use draw 0 as pass@1 and one group of four. Metrics and aggregation denominators are not interchangeable.

The reported LEDGAR recipe uses per-case accuracy, response reflection batch3/seed42, judge batch16/seed43, compact JSON grammar, and greedy 512-token judges. CaseHOLD, LAR-ECHR, SARA and LEXam use judge reflection batch4 to fit the proposer context. NPD/PARCEL used the adapted cohort recipe, MAE/macro-F1, sampled 1024-token responses and judges, and sampled 128-token answer forcing on length failures. See each YAML for complete settings.

PARCEL optimization worsened both DEV and TEST scores. NPD remains unstable: its TEST optimized individual has 60 failures (88.28% coverage), while optimized median@4 excludes five failures. Low-coverage, valid-only MAE is **not reliable evidence of improvement**. All parseable finite values, including extreme numbers, remain in the metric. Failures are never converted to arbitrary numbers. Exact [failure/coverage counts](results/sequential/final-DEV-TEST-table.csv) and [forcing attempts/recoveries](results/sequential/final-forcing-counts.csv) are published; reported failure counts unavailable in summary sources are not inferred as zero.

## Exposure and reproduction limits

- LEDGAR judge cap512 was chosen using a TEST comparison against cap1024; results are exploratory.
- CaseHOLD reported DEV512 included two cases also in FIT. The explicit public recipe removes those cases (DEV510); a clean rerun is not the same DEV evaluation.
- LAR-ECHR has 58/98 DEV questions overlapping FIT/SELECT; BarExamQA has 20/123.
- SARA and LEXam retain reported DEV/TEST halves derived from source TEST.
- NPD had earlier optimization exposure to official TEST through reshuffled partitions.
- PARCEL TEST contains 25 earlier FIT, 8 SELECT, 70 DEV and 81 TEST cases. All 34 opinions overlap DEV/TEST.
- Endpoint behavior, inference kernels and fresh optimization may change outcomes. A new run generates its own response banks; reported scores do not certify new predictions.

## Run or evaluate

See [README](README.md) for the suite command. `--saved-prompts` imports the final instructions and evaluates them without running GEPA. Imports are labeled as saved, bind to verified cohort inputs, and never claim a new optimization. TEST readouts use frozen prompts and never select configurations.

The runnable configs use vLLM 0.33.0 and `structured_outputs`. The reported scores were not rerun for this serving migration: NPD/PARCEL used Phi on vLLM 0.10.0 with `guided_json` and Gemma on 0.33.0. Saved prompts can be evaluated with the current configs, but identical numerical scores are not established.
