# All-component joint follow-up: paused checkpoint

This extends task3 using the same all-component config on all datasets. All inference uses frozen Phi-4-mini. Prompt-change and performance outcomes are separate. Programs are selected on SELECT; DEV/TEST outcomes do not guide revisions.

| Dataset | Optimization | Selected prompt changes | DEV judge | TEST judge |
|---|---|---|---:|---:|
| ledgar | completed_audited | persona0, persona1, persona2, persona3, aggregator | 56.79% | 54.10% |
| casehold | completed_audited | None | 62.65% | 58.20% |
| lar_echr | failed_context | Unavailable | Unavailable | Unavailable |
| sara_entailment | interrupted_by_user | Unavailable | Unavailable | Unavailable |
| barexamqa_closed_book | completed_audited | persona0, persona1, persona2, persona3, aggregator | 53.46% | 33.47% |
| lexam | completed_audited | persona0, persona1, persona2, persona3, aggregator | Unavailable | Unavailable |

LAR failed proposer context at1480 student/75 proposer calls. Its separately prepared minibatch1 follow-up never started. SARA optimization and LEXam readout were interrupted by direct user pause (actual exit-15); partials remain local and must not silently resume. No result is inferred from partials.

Full selected prompts, audits, call counts, source plan hashes and status: `checkpoint.json`.
