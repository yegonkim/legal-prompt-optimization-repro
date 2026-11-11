# Task 1: fixed full-reasoning response and judge optimization

Frozen Phi-4-mini inference. Response optimization seed42; judge optimization seed43, reflection minibatch16, SELECT chooses prompts. Generator cap1024, judge cap512. Judge receives original question plus all four complete response texts. No per-dataset rescue tuning.

| Dataset | Split | Ordinary pass@1 | Ordinary vote@4 | Optimized pass@1 | Optimized vote@4 | Optimized + judge |
|---|---|---:|---:|---:|---:|---:|
| LEDGAR | DEV | 50.79% | 52.29% | 58.53% | 60.74% | 62.06% |
| LEDGAR | TEST | 47.13% | 49.51% | 56.32% | 58.64% | 60.21% |
| CaseHOLD | DEV | 60.86% | 61.72% | 61.63% | 63.53% | Unavailable |
| CaseHOLD | TEST | 55.88% | 56.10% | 56.93% | 57.32% | Unavailable |
| LAR-ECHR | DEV | 43.81% | 43.62% | 43.81% | 43.62% | Unavailable |
| LAR-ECHR | TEST | 45.79% | 46.67% | 45.79% | 46.67% | Unavailable |
| Official SARA | DEV | 67.00% | 70.50% | 64.38% | 66.00% | Unavailable |
| Official SARA | TEST | 66.38% | 68.50% | 66.38% | 68.00% | Unavailable |
| LEXam | DEV | 21.28% | 21.88% | 25.89% | 28.37% | Unavailable |
| LEXam | TEST | 22.31% | 23.34% | 24.93% | 26.61% | Unavailable |
| BarExamQA | DEV | 49.75% | 51.63% | 49.75% | 51.63% | 47.56% |
| BarExamQA | TEST | 35.17% | 35.59% | 35.17% | 35.59% | 40.25% |

CaseHOLD, LAR-ECHR, official full-section SARA and LEXam judge optimization exceeded the fixed proposer context window. Failures/partials preserved; no clipping, batch reduction, or seed fallback called a successful optimized judge. Their actor-only reporting audits passed. Bar and LEDGAR full judge reporting audits passed.

DEV uses official validation where available: LEDGAR512, CaseHOLD512 after original two-tokenizer1800-token filter, LAR98, Bar123. SARA50/TEST50 and LEX512/TEST512 retain historical TEST-derived halves. TEST membership unchanged. Official SARA uses authors facts/questions and complete dataset-curated relevant statute section. DEV question-text overlap with FIT/SELECT: Case2/512, LAR58/98, Bar20/123; these are exposed development cohorts, not fresh heldout evidence.

Controlled judge output cap comparison: 512 DEV1271/2048, TEST1233/2048;1024 DEV1268/2048, TEST1229/2048. User explicitly requested binary TEST-cap selection; retained512. Report this as TEST-selected/exploratory, not independent heldout confirmation. No further TEST-driven optimization.

Exact source records: configs/migration/LED_full_reasoning_user_accepted_reflect16_seed43_20261110_v1.json; LED_judge_cap_final_TEST_comparison_20261110_v1.json; fixed_transfer_{casehold,lar_echr,sara_entailment,lexam}_actor_DEV_TEST_20261110_v2_results.json; fixed_full_reasoning_bar_officialDEV_TEST_20261110_v1_results.json. Raw artifacts remain outside Git.
