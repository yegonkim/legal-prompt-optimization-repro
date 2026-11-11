# Task 1: completed minibatch-4 follow-up

The user authorized reducing judge reflection minibatches from 16 to 4 for the four datasets whose original proposer inputs exceeded context. All other optimization settings stayed fixed. Original failures and the original report are preserved. Every readout exited 0 and passed audit. Judge input contains all four complete responses, including reasoning.

| Dataset | Split | Ordinary pass@1 | Ordinary vote@4 | Optimized pass@1 | Optimized vote@4 | Optimized + judge |
|---|---|---:|---:|---:|---:|---:|
| CaseHOLD | DEV | 60.86% | 61.72% | 61.63% | 63.53% | 63.33% |
| CaseHOLD | TEST | 55.88% | 56.10% | 56.93% | 57.32% | 58.30% |
| LAR-ECHR | DEV | 43.81% | 43.62% | 43.81% | 43.62% | 43.37% |
| LAR-ECHR | TEST | 45.79% | 46.67% | 45.79% | 46.67% | 51.50% |
| Official SARA | DEV | 67.00% | 70.50% | 64.38% | 66.00% | 66.00% |
| Official SARA | TEST | 66.38% | 68.50% | 66.38% | 68.00% | 60.50% |
| LEXam | DEV | 21.28% | 21.88% | 25.89% | 28.37% | 26.46% |
| LEXam | TEST | 22.31% | 23.34% | 24.93% | 26.61% | 25.49% |

These are fixed-winner reporting results; TEST outcomes did not select or revise these follow-up methods. The previously documented official-DEV membership, FIT/SELECT overlaps and historical TEST exposure still apply. LEDGAR and BarExamQA retain the minibatch-16 results in the original task1 report.

Exact result files: `configs/migration/fixed_full_reasoning_{casehold,lar_echr,sara_entailment,lexam}_reflect4_DEV_TEST_20261110_v1_results.json`. Raw generations remain outside Git.
