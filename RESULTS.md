# Recorded results

These are immutable records from the original audited experiments. The portable runner has CPU contract/integration and six-dataset input-restoration verification, but has not itself reproduced these GPU scores. Exact selected instructions are stored in `results/<dataset>/*_winner.json`.

## Task2: neutral joint, round-robin mutation

| Dataset | DEV individual | DEV vote@4 | DEV judge | TEST individual | TEST vote@4 | TEST judge |
|---|---:|---:|---:|---:|---:|---:|
| ledgar | 49.06% | 56.93% | 53.66% | 44.01% | 53.17% | 49.32% |
| casehold | 61.12% | 61.52% | 61.91% | 56.48% | 57.13% | 58.64% |
| lar_echr | 38.71% | 39.03% | 37.76% | 44.29% | 45.83% | 43.33% |
| sara_entailment | 65.38% | 70.00% | 68.50% | 64.88% | 66.50% | 64.50% |
| barexamqa_closed_book | 48.07% | 49.39% | 49.80% | 36.33% | 40.25% | 40.68% |
| lexam | 24.17% | 25.63% | 24.12% | 24.44% | 26.56% | 24.90% |

All six selected judges retained their neutral initialization in these round-robin runs. Some or all generators changed, and CaseHOLD selected the entire seed program unchanged. Prompt changes were not a task2 success gate.

## Task1: response then full-reasoning judge

Complete baseline/sequential tables are in `results/reports/TASK1_FULL_REASONING_FIXED_RECIPE_REPORT_20261110.md` and the separate minibatch4 follow-up report. Original minibatch16 proposer-context failures are preserved; follow-up results must not be mistaken for successes of that original setting. The sequential response and judge winner JSONs here are the audited completed follow-up selections.

LEDGAR sequential full-reasoning judge:62.060546875%DEV and60.205078125%TEST. The matched optimized vote@4 scores were60.7421875%DEV and58.642578125%TEST. Its512-token judge exceeded the1024-token reoptimized judge on both cohorts; the binary TEST-cap choice is explicitly exploratory.

## Task3: all-component mutation on BarExamQA

Neutral initialization; GEPA all-component/Pareto/minibatch3/seed42/no merge. All five instructions in the SELECT-selected program changed substantively. SELECT28/64→33/64 (43.75%→51.5625%);4715 student and85 proposer calls. Original generation exit0/auditPASS. There was no DEV/TEST readout of this particular task3 program when these records were exported. No claim of its cross-dataset efficacy.

See `results/task3_all_five_changed_final_20261111_v1.json` and `results/task3_all_five_changed_prompts.txt`.

## Verify and extend

`python -m legal_prompt_optimization verify-bundle` verifies shipped checksums, not hidden raw generations. The source study audits and source-file hashes are archived; rerun and audit the portable code locally for new evidence. Add future runs under new names with their actual exits/audits and no overwrite. Keep failures and partials locally; never silently resume an old budget.
