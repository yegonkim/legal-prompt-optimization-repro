# Machine handoff — paused 2026-11-11

## Start here

Paths under `configs/migration/`, `dev_utils/` and historical `docs/` below refer to the research archive unless explicitly identified as private-package paths. The private-package checkpoint is `results/all_component_followup_20261111/checkpoint.json`.

**Experiments are paused by direct user request. Do not launch studies, resume interrupted attempts, or install schedulers until the user explicitly resumes work.** GPU watcher and daily Git cron entries were removed; Astra was already absent. Both GPUs were verified idle after our jobs and orphan workers stopped. The original machine's raw artifacts remain intact.

Two repositories:

- Research archive: https://github.com/yegonkim/msv_legal (branch `auto`). Original tasks1–3 were pushed through `1d5f3af`; this handoff accompanies the later curated checkpoint upload.
- Small standalone package: https://github.com/yegonkim/legal-prompt-optimization-repro (private, branch `main`). Usability improvements through `c5caae1`; this handoff upload adds later result evidence. GitHub access to the private repository is required.

Original local paths were `/workspace/msv_legal` and `/workspace/legal-prompt-optimization-repro`; these are historical paths, not requirements on a new host. Do not globally rewrite old provenance hashes/paths. Create new versioned run configs for the new host.

## User's actual priorities

Frozen `microsoft/Phi-4-mini-instruct` for all inference; no fine-tuning. Prompt optimization uses FIT feedback, SELECT selects winners, DEV/TEST readouts evaluate frozen programs. The user retired the old +2pp goal and gate for these tasks. Old scheduler messages repeating that goal are stale. Do not use TEST outcomes to revise methods.

Cleanliness, ease of use and self-contained operation take priority over exact historical reproduction. Fresh candidate banks are the correct default. Explicitly tell the user when dependency upgrades, fresh banks, decoding changes or different data membership reduce comparability. Detailed replay audit is optional in the small repo; basic checks run automatically. `evaluate --baselines` optionally produces ordinary pass@1/vote@4, optimized pass@1/vote@4 and optimized+judge from matched candidates.

## Completed original tasks

1. Response optimization then a judge that receives all four full raw responses, including reasoning. LEDGAR accepted512-token judge: DEV62.060546875%, TEST60.205078125%. Fresh1024-cap judge was worse on both;512 retained. Historical binary cap selection used TEST by direct user request and is marked exploratory. Fixed minibatch16 transfer exceeded proposer context for four datasets; authorized minibatch4 follow-ups completed and were reported separately. See `docs/TASK1_FULL_REASONING_FIXED_RECIPE_REPORT_20261110.md` and `docs/TASK1_REFLECT4_FOLLOWUP_REPORT_20261111.md`.
2. Neutral-start joint four-generator+judge optimization across six datasets, one shared round-robin config; all optimizations and DEV/TEST reports completed. All six selected judges remained neutral. See `docs/TASK2_NEUTRAL_JOINT_REPORT_20261111.md`, curated `neutral_joint_report_20261111_v1.json`, and selected prompt exports.
3. Find a configuration with substantive changes to all five SELECT-selected prompts. BarExamQA succeeded: all-component GEPA/Pareto/reflection minibatch3/seed42/no merge; SELECT28/64→33/64;4715 student/85 proposer calls. See `docs/TASK3_ALL_FIVE_CHANGED_REPORT_20261111.md` and prompts TXT.

## Later requested research: status at pause

The user requested applying task3's exact config to all datasets with DEV/TEST readouts, and finding a per-dataset setting whose selected five prompts change. Optimize cap9911 student/85 proposer/5.5h; actor1024 sampled(.7/.8/top-k20), judge512 greedy, full four raw responses. Readouts four independent trials,16 actors+4 judge calls/question,4h cap. Completed Bar optimization was reused, never duplicated.

| Dataset | All-component optimization | Five prompts changed? | DEV judge | TEST judge |
|---|---|---|---:|---:|
| LEDGAR | Completed | Yes |56.79%|54.10%|
| CaseHOLD | Completed, selected neutral seed |No|62.65%|58.20%|
| BarExamQA | Completed original task3 |Yes|53.46%|33.47%|
| LEXam | Completed |Yes|Interrupted|Interrupted|
| LAR-ECHR | Failed proposer context |No completed winner|Not run|Not run|
| Official SARA | Interrupted by user |No completed winner|Not run|Not run|

Exact unrounded scores, source hashes, selected instructions, audits, failure and interruption counts are in `configs/migration/all_component_transfer_checkpoint_20261111_v1.json`. See `docs/ALL_COMPONENT_TRANSFER_CHECKPOINT_20261111.md`.

LAR fixed minibatch3 failed at1480 student/75 proposer calls, actual exit1. Separate minibatch1 follow-up was prepared but **never started**. This is a different config and must be labeled accordingly. CaseHOLD needs a new FIT/SELECT-only setting search if work resumes. LEXam readout and SARA optimization were explicitly terminated, both actual exit-15; never silently resume them or call partials final results.

Future work after explicit resume: finish separately versioned remaining reports/searches, update the private repo with final results, then run end-to-end checks using the **private repo code**, not old research scripts. None of those full GPU reproducibility checks has been completed.

## New-stack compatibility work — GPU comparison NOT RUN

CPU regression with the actual pinned Phi tokenizer and no hook: XGrammar0.2.7 failed mask/matcher consistency inside unfinished JSON;0.2.8 and0.2.9 passed all8 checks. See `dev_utils/check_xgrammar_stop_mask_20261111_v1.py` and `configs/migration/xgrammar_stop_mask_CPU_comparison_20261111_v1.json`. Upstream fix: https://github.com/mlc-ai/xgrammar/pull/905 .

Old research stack: vLLM0.31.0, torch2.13.0, XGrammar0.2.7, Transformers5.17.0. The Phi EOS compiler alignment hook is needed for that stack. It aligns stop IDs199999/200020. Keep the default private-repo dependencies/hook until a clean newer stack has actually passed GPU checks.

An **isolated** new environment was built on the original machine: `outputs/environments/vllm033-compat-20261111-v1/bin/python`; vLLM0.33.0,torch2.14.0+cu130,XGrammar0.2.8,Transformers5.17.0. Dependency closure198packages and pip check passed; old210package metadata fingerprints unchanged; CPU imports left CUDA uninitialized and loaded no hook. Its5.1GB environment is not in Git. Small version/wheel-hash readiness metadata is curated separately; reconstruct an environment rather than copying it. The original environment uses read-only inheritance of old dependencies, so it is not portable by directory copy.

Frozen comparison plan `same_seed_stack_comparison_20261111_v1.json`: two FIT rows/dataset, ordinary/sequential/joint profiles,144actors+36judges=180calls/stack. Run old patched baseline, old repeat and new clean stack on the **same physical GPU**, total540calls,1h cap per stack. Judges must use identical baseline-first candidate frames. Compare exact token IDs, text, parsed answers, validity and finish reason; baseline repeat measures existing nondeterminism. Sample equality does not prove all historical scores transfer.

Prepared runner is curated as `dev_utils/run_same_seed_stack_comparison_20261111_v1.py`. It does **not** acquire GPU leases; acquire an exclusive lease and check vacancy externally. Supply `--repo` pointing to the private package checkout. All GPU comparison outputs are absent: this test has NOT RUN. Original input JSON contains raw FIT text and is deliberately outside Git at `outputs/research_preparation/same_seed_stack_comparison_20261111_v1/inputs.json`; reconstruct a separately versioned equivalent from restored FIT cohorts/selected prompts, or securely transfer that existing small file outside Git and verify its recorded hash. Do not claim the old input-byte hash if rebuilding different metadata.

## Data and package caveats

Use the small repo's README and DATA.md: `prepare` downloads pinned public sources and restores shipped cohort IDs/hashes, `verify-data` checks them. Official SARA is the authors' complete dataset-curated relevant statute section, not LegalBench's truncated version. Official DEV is used where available. CaseHOLD length filter, two malformed Bar exclusions, LEXam research-ID exclusion and historical exposure are recorded. LAR official DEV and some other DEV cohorts overlap earlier training question text; do not claim fresh holdout evidence.

**Still missing:** an accurate complete pipeline that regenerates manifest membership from the original sorting/random sampling/filtering history. Current code restores shipped manifests. Do not invent a universal split recipe or erase historical exceptions.

The small repo's22CPU tests and fresh-clone bundle checks passed; its GPU numerical results were archived from original runs, not reproduced by a completed end-to-end run of the new package.

## What Git deliberately does not carry

No model weights/checkpoints, installed environments, source/question data, raw sampled generations, reflection logs, caches, credentials or host-local scheduler state. Every curated file is below1MiB. Original full source-study artifacts can be separately backed up/transferred when needed; a clone is sufficient to read results and run fresh self-contained package workflows, but does not recreate raw historical trajectories or interrupted attempts. Preserve any existing artifacts; do not delete them as part of migration.
