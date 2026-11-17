# Data and membership

Each row is a normalized JSON object. A synthetic illustration (not a benchmark item):

```json
{"record_id":"synthetic:1","question":"Choose the supported alternative. A. Alpha B. Beta","public_labels":["A","B"],"answer":"A","reference_feedback":"Synthetic training feedback."}
```

`reference_feedback` is required for response FIT/SELECT preparation and optional for readout rows. It only enters FIT reflection. `source_record_id`, if present, affects joint actor seeds; do not discard or reformat it. All option labels are public inputs. `answer` and `reference_feedback` are never inserted into model inference prompts.

| Dataset | FIT | SELECT | DEV | TEST | Upstream |
|---|---:|---:|---:|---:|---|
| LEDGAR |128|64|512|512|`coastalcph/lex_glue`, config LEDGAR|
| CaseHOLD |128|64|512|512|`casehold/casehold`, original all/CSV source|
| LAR-ECHR |40|20|98|150|`AUEB-NLP/lar-echr`|
| Official SARA entailment |64|32|50|50|JHU authors archive, complete dataset-curated relevant statute section|
| BarExamQA closed book |128|64|123|59|`reglab/barexam_qa`; public inference excludes gold_passage|
| LEXam |128|64|512|512|`LEXam-Benchmark/LEXam`, English4/8/16/32-choice sources|

Pinned revisions live in `legal_prompt_optimization/data.py` and the manifests. SARA archive URL: `https://nlp.jhu.edu/law/sara/sara.tar.gz`; SHA256 `e4f1b845016fc38b95b702bbbb27a5c9af3eadfa856ba585662b8b2cd71c187c`. The authors simplify selected tax-code sections; “complete section” does not mean the entire US Code.

Manifests preserve exact ordered IDs for all cohorts, with SHA256 of canonical JSON containing record_id, question, answer, public_labels, source_record_id when present and reference_feedback when present. They contain no raw question or feedback text or gold labels. Restoration verifies every row. JSON whitespace and non-input metadata need not match the original file-byte hashes; the original file hashes remain recorded for provenance.

CaseHOLD's existing cohort uses a <=1800-public-question-token eligibility filter under the pinned Phi and historical Qwen tokenizers. The frozen IDs implement that membership directly, so restoration does not download the historical Qwen model/tokenizer. Bar excludes the two originally malformed source items, `mbe_494` and `mbe_1121`, before validation row indexing.

DEV now uses official validation where available. CaseHOLD, LAR and Bar official validation overlap historical FIT/SELECT question text by2/512,58/98 and20/123 respectively; exposed development results must not be described as fresh held-out evidence. SARA and LEXam retain historically TEST-derived DEV/TEST halves; LEXam's official TEST source was used as research data, with the recorded research-training membership excluded when those cohorts were frozen. Earlier data exposure is not erased by restoration.

FIT and SELECT IDs must be disjoint within each run. Only FIT trajectories supply GEPA feedback. SELECT chooses the winner; evaluation freezes that winner before DEV/TEST inference. The optimizer verifies only FIT/SELECT files and never opens DEV/TEST rows. No fixed +2pp gate is imposed by this reproduction package: the user authorized fixed-program DEV/TEST reporting. TEST outcomes do not choose or revise methods, except the explicitly documented historical512-vs1024 binary cap choice.

Public data access and dataset licenses remain the user's responsibility. This code repository redistributes no dataset texts or model weights. Cache and restored data directories are ignored.

Public builder correction: the current packaged CaseHOLD manifest removes the two DEV/FIT-overlapping cases and produces DEV510. The table above describes the archived experiment, not the corrected public rerun.
