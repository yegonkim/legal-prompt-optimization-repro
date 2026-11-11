# Task 2: neutral-start joint optimization

All six programs started with four identical ordinary instructions and one neutral judge. GEPA could edit all five components; the judge received all four complete response texts. Shared optimization settings: seed42, Pareto, round-robin components, reflection minibatch3, no merge, at most9,911 student calls/88 proposer calls/5.5hours. Frozen Phi-4-mini inference; Gemma proposer receives FIT feedback only.

Readout: four independent trials per question, four generators per trial,1024-token sampled actors and512-token greedy judge. Individual and voting scores use the same candidates as the judge. Readout outcomes do not select or revise programs.

| Dataset | Split | Joint individual | Joint vote@4 | Joint judge |
|---|---|---:|---:|---:|
| LEDGAR | DEV | 49.06% | 56.93% | 53.66% |
| LEDGAR | TEST | 44.01% | 53.17% | 49.32% |
| CaseHOLD | DEV | 61.12% | 61.52% | 61.91% |
| CaseHOLD | TEST | 56.48% | 57.13% | 58.64% |
| LAR-ECHR | DEV | 38.71% | 39.03% | 37.76% |
| LAR-ECHR | TEST | 44.29% | 45.83% | 43.33% |
| Official SARA | DEV | 65.38% | 70.00% | 68.50% |
| Official SARA | TEST | 64.88% | 66.50% | 64.50% |
| BarExamQA | DEV | 48.07% | 49.39% | 49.80% |
| BarExamQA | TEST | 36.33% | 40.25% | 40.68% |
| LEXam | DEV | 24.17% | 25.63% | 24.12% |
| LEXam | TEST | 24.44% | 26.56% | 24.90% |

DEV membership and previously documented exposure apply unchanged: official validation where available; historical TEST-derived halves for SARA and LEXam. Existing FIT/SELECT question overlaps mean some DEV cohorts are exposed development evidence. Do not interpret these as fresh holdouts.

Selected programs and exact audits, per-generator metrics, invalid and length counts are recorded in `configs/migration/neutral_joint_report_20261111_v1.json`. One SARA DEV response and six LEXam DEV/four LEXam TEST responses hit the token cap and were invalid; all remain included. Raw questions, responses and logs stay outside Git.

Status: all six readouts audited.
