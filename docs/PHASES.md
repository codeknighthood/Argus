# ARGUS: phase-by-phase development log

One phase at a time. A phase is closed only when its "done when" line is true and its tests pass.
Each phase below says what was built, how to run it, and what is still waiting on a real run on your machine.

Status key: **VERIFIED** = run on your machine and checked. **BUILT** = code written and unit-tested here,
with a fake model, but not yet run by you on real data. **NOT STARTED**.

| Phase | Synopsis | Status |
|---|---|---|
| 0. Setup and decisions | all | VERIFIED |
| 1. Data and preprocessing | 5.1, 5.2 | BUILT (needs the real corpus) |
| 2. Stance detection | 5.5, 5.6 | BUILT (needs a real run) |
| 3. Argument extraction | 5.3 | BUILT (needs a real run) |
| 4. Argument relations | 5.4 | BUILT (needs a real run) |
| 5. Integrated pipeline | 5.6, 5.7 | BUILT (needs a real run) |
| 6. Evaluation and error analysis | 5.8 | BUILT (needs a real run) |
| 7. Demo and final package | - | BUILT (needs a run on your machine) |

Decisions fixed so far: Ollama only (no API keys), model `qwen2.5:7b`, stance labels Favor / Against / Neutral,
primary dataset VAST (replaced the UKP corpus, which needs an access form; see Phase 1 note).

---

## Phase 0: Setup and decisions  (VERIFIED)

- **Built:** project layout, `config.yaml`, Ollama client with schema-constrained JSON, validation, retries and
  an on-disk cache (`argus/llm.py`), the output schema (`argus/schema.py`), first stance prompt.
- **Run:** `python scripts/smoke_test.py`
- **Done when:** one local model call returns valid JSON for a sample statement. Confirmed on your machine
  (qwen2.5:7b returned Favor and Against with sensible reasons).

## Phase 1: Data and preprocessing  (BUILT)

- **Built:** `argus/data.py` (common `Record`, UKP loader, cleaning, de-duplication, balanced dev/test samples),
  `argus/segment.py` (transcript to sentences with speaker and carried-over context), inspect and prepare scripts.
- **Run:** `python scripts/inspect_data.py`, then `python scripts/prepare_data.py`
- **Done when:** a loader turns raw data into `(target, statement, context, gold_label)` records.
- **Still open:** run it on the real corpus and confirm the column names matched (`inspect_data.py` output).

## Phase 2: Stance detection  (BUILT)

- **Built:**
  - `argus/stance.py`: three prompting strategies, `zero_shot`, `few_shot` (six worked examples on topics
    outside the benchmark, so there is no leakage) and `cot` (reasoning first, then the label).
  - `argus/metrics.py`: accuracy, per-class precision/recall/F1, macro and weighted F1, confusion matrix,
    Cohen's kappa. Plain Python, checked against hand-computed values.
  - `argus/evaluation.py`: runs a stage over a dataset; a failed call becomes `ERROR` and counts as wrong.
  - `scripts/run_stance.py`, `argus/prompting.py`, versioned prompt files in `argus/prompts/`.
- **Run:**
  ```
  python scripts/run_stance.py --strategy zero_shot --limit 60     # quick first look
  python scripts/run_stance.py --strategy zero_shot                # full dev sample
  python scripts/run_stance.py --strategy few_shot
  python scripts/run_stance.py --strategy cot
  ```
  Writes `results/stance_<strategy>_<split>.jsonl` (one row per statement) and `.json` (the scores).
- **Done when:** zero-shot and few-shot prompts are scored with accuracy, precision, recall and F1.
- **Rule for honest numbers:** tune prompts on `dev` only. Run `--split test` once per strategy for the numbers
  you report.
- **Tests:** `python -m unittest discover -s tests -t . -v` (41 tests, no Ollama needed).
- **Still open:** the first real scores. Nothing about model quality is known until you run it.

## Phase 3: Argument extraction  (BUILT)

- **Built:**
  - `argus/extract.py` and `argus/prompts/extract_v1.txt`: for one sentence, the model labels it Claim, Premise
    or None and rewrites the argument as a short self-contained sentence, with pronouns resolved from the context.
    That restatement is what Phase 4 compares. Contradictory output is tidied (None always has an empty
    restatement; an argument with no restatement falls back to the original sentence).
  - `argus/evaluation.py`: `predict_arguments` scores Argument vs NoArgument against the UKP gold labels
    (Claim and Premise both count as Argument).
  - Annotation tooling, because UKP has **no claim/premise labels**: `argus/annotation.py`,
    `scripts/make_annotation_sheet.py`, `scripts/run_annotated_eval.py`, `scripts/annotator_agreement.py`,
    the labelling rules in `docs/ANNOTATION_GUIDE.md`, and a 23-sentence sample transcript in
    `data/sample/` (written for testing, not a real debate).
- **Run:**
  ```
  python scripts/run_extraction.py --limit 60            # Argument vs NoArgument on UKP dev
  python scripts/make_annotation_sheet.py --kind components   # then fill gold_component in Excel
  python scripts/run_annotated_eval.py --target "the minimum wage"
  python scripts/annotator_agreement.py A.csv B.csv      # two annotators, Cohen's kappa
  ```
- **Done when:** the extraction prompt outputs typed components and is scored on an annotated set.
- **Tests:** 61 at the end of this phase, no Ollama needed. The scripts were also run end to end against a fake Ollama server.
- **Still open:**
  - Real scores. Nothing about model quality is known until you run it.
  - A transcript annotated by hand (ideally two people) for the Claim / Premise / None score.
  - A real transcript source. The sample is only for trying the tools.
  - UKP's "argument" means "argues for or against the topic", which is close to, but not identical with,
    Claim or Premise. Expect some disagreement and discuss it in the error analysis.

## Phase 4: Argument relations  (BUILT)

- **Built:**
  - `argus/relations.py` and `argus/prompts/relation_v1.txt`: given an earlier argument A and a later argument B
    (with their speakers), the model says whether B Supports, Attacks or is unrelated (None) to A. The prompt judges
    B toward A, not toward the policy, so two arguments that are both against the policy can still support each other.
  - `find_relation`: links an argument to the **nearest** of the previous N arguments (default 3) that it supports or
    attacks, checking nearest first and stopping at the first link. This follows how debates flow and caps the cost
    at N model calls per argument. Its limitation (it can miss a link to something further back when a nearer
    argument also connects) is stated in the module and belongs in the report's limitations.
  - Annotation and scoring: `make_annotation_sheet.py` now also writes `pairs.csv` (one row per A/B pair),
    `scripts/run_relation_eval.py` scores the model against your labels (precision, recall, F1, kappa, and accuracy
    split by same-speaker vs different-speaker pairs), `annotator_agreement.py --column gold_relation` gives kappa
    between two annotators. Relation rules added to `docs/ANNOTATION_GUIDE.md`.
- **Run:**
  ```
  python scripts/make_annotation_sheet.py --kind pairs   # fill gold_relation in Excel
  python scripts/run_relation_eval.py --target "the minimum wage"
  python scripts/annotator_agreement.py A_pairs.csv B_pairs.csv --column gold_relation
  ```
- **Done when:** relation output is scored on annotated pairs.
- **Tests:** 83 in total, no Ollama needed. The scripts were also run end to end against a fake Ollama server.
- **Still open:**
  - Real scores, and a hand-annotated pairs sheet (ideally two annotators).
  - The relation step is scored on the sentences as written. Inside the full pipeline (Phase 5) it works on the
    restated arguments from Phase 3, so end-to-end relation quality is checked in Phase 6.
  - Each pair is judged alone. Chains of reasoning (A supports B supports C) are not modelled; this is a stated
    limit of the approach.

## Phase 5: Integrated pipeline  (BUILT)

- **Built:**
  - `argus/pipeline.py`: `ArgusPipeline` chains segmentation, extraction (Phase 3), stance (Phase 2) and relations
    (Phase 4) and returns one `ArgumentRecord` per sentence, the structured output of synopsis 5.7. The LLM client is
    shared by every stage, which matches synopsis 5.6 (the LLM is not a final step; it powers all of them).
  - **A failed model call costs one field of one row, never the run.** Each stage is wrapped separately, the row
    records what failed in its `error` column, and the other fields are still filled. Tested for each stage.
  - `argus/output.py`: JSON (with the settings used, so a run can be reproduced), CSV (UTF-8 with BOM for Excel; the
    nine synopsis 5.7 columns first), a compact Markdown table, and headline counts.
  - `scripts/run_pipeline.py`: transcript in, `.json` + `.csv` + `.md` out.
  - Settings live in a new `pipeline:` section of `config.yaml` (stance strategy, context window, relation window,
    stance on all sentences or only arguments, minimum sentence length); command-line options override them.
  - Plain error messages instead of tracebacks when Ollama is not running, in every script.
- **Run:**
  ```
  python scripts/run_pipeline.py                                    # the sample transcript
  python scripts/run_pipeline.py --limit-segments 8                 # quick try
  python scripts/run_pipeline.py --transcript my_debate.txt --target "the minimum wage"
  ```
- **Done when:** a raw transcript goes in and a JSON/table of structured results comes out.
- **Tests:** 116 in total, no Ollama needed. The script was also run end to end against a fake Ollama server.
- **Choices to know about (for the report):**
  - `stance_strategy` defaults to `few_shot` but that is a placeholder: Phase 6 compares the three strategies and
    the config should then be set to the best one.
  - Cost: each sentence needs one extraction call, one stance call and up to `relation_window` relation calls, so
    a transcript of 100 sentences means several hundred model calls. Answers are cached, so re-runs are free. Not
    yet timed on your GPU.
  - `stance_on: arguments` skips the stance call for sentences known not to be arguments and leaves their stance
    empty (not "Neutral"), so a skipped row is never mistaken for a judged one.
  - Sentences shorter than `min_tokens` (default 3 words) are dropped, so "I disagree." disappears. The annotation
    sheets use the same default so sentence numbers line up (there is a test for this). Set `min_tokens: 2` in
    `config.yaml` and pass `--min-tokens 2` to `make_annotation_sheet.py` to keep two-word sentences in both.
  - The relation stage works on the restated arguments from extraction, not the raw sentences.
- **Still open:** a real run on your machine. Nothing about the quality of the combined output is known until
  then; Phase 6 measures it.

## Phase 6: Evaluation and error analysis  (BUILT)

- **Built:**
  - `scripts/evaluate.py` and `argus/compare.py`: run zero-shot, few-shot and chain-of-thought (and optionally
    several models) on the same sentences. Every run gets accuracy, per-label precision / recall / F1, macro and
    weighted F1, a confusion matrix and per-topic accuracy.
  - `argus/metrics.py`: a 95% bootstrap confidence interval (fixed seed, so repeatable) and an exact McNemar test, so
    the report can say whether one strategy is really better or just lucky on this sample. Plain Python, checked by hand.
  - `argus/report.py`: `results/evaluation/summary.md`, a report with 9 numbered sections plus a "How far to trust
    this" section, and `summary.json` with the same numbers.
  - `argus/analysis.py`: automatic cues (the shape of each mistake, and sentence features linked to errors) and the
    seven human-tagged causes. `errors_<run>_<split>.csv` lists every mistake in a fixed random order with a blank
    `error_type` column to fill in; `scripts/tally_errors.py` counts the causes. Procedure: `docs/ERROR_ANALYSIS.md`.
  - Self-consistency (`--consistency-runs 3`): repeats the best run at temperature 0.7 and reports how often the
    label changes between runs. Uses a cache salt so each run really asks the model again (the old cache keys stay valid).
  - `scripts/evaluate_pipeline.py` and `argus/pipeline_eval.py`: run the WHOLE pipeline on an annotated transcript and
    score components and relations. For every missed link it says why (argument not detected, outside the relation
    window, linked to a nearer argument, model judged unrelated ...). It checks that your sheets match the transcript.
  - `pipeline.stance_on: none` skips the stance stage, since a transcript has no stance labels.
  - `annotator_agreement.py --column error_type` gives kappa between two people tagging causes.
- **Run:**
  ```
  python scripts/evaluate.py --limit 60                         # quick look on dev
  python scripts/evaluate.py                                    # all three strategies, whole dev sample
  python scripts/evaluate.py --consistency-runs 3               # plus answer stability
  python scripts/evaluate.py --split test --strategies few_shot # the numbers to report (once)
  python scripts/tally_errors.py results/evaluation/errors_<run>_test.csv
  python scripts/evaluate_pipeline.py --target "the minimum wage"   # after annotating (Phases 3 and 4)
  ```
- **Done when:** the strategies are compared with intervals and significance tests, mistakes are analysed by cause,
  and the pipeline is scored end to end on an annotated transcript.
- **Tests:** 167 in total (51 new), no Ollama needed. All three scripts were also run end to end against a fake
  Ollama server.
- **Choices to know about (for the report):**
  - Choose the strategy on **dev**, then score **test** once. The script prints a reminder on the test split.
  - With roughly 700 test sentences, a gap of about 3 points between strategies is usually inside the noise; the
    McNemar test says so explicitly.
  - Automatic cues are leads, not explanations. The causes in the report must come from your own tagging.
  - `--limit` takes a random subset, so small trial runs are not just the first topic in the file.
- **Still open:** everything that needs the real model. No real score exists yet. Phase 6 is the machinery; the
  numbers come from your runs.
- **Rebuilt:** Phase 6 was rewritten from scratch after the first build was lost. The design and tests are the same.

## Phase 7: Demo and final package  (BUILT)

- **Built:** `app.py` (Streamlit demo: condensed view, argument map, full table, downloads), `argus/demo.py` (the logic
  behind it, tested without a browser; repeated points are merged by word overlap), `scripts/check_setup.py` and
  `argus/setup_check.py` (diagnose Python, packages, Ollama, model and data, with the exact fix for each), `scripts/quickstart.py`
  (runs the workflow in order), `INSTALL.md` (the single start-to-finish guide), `requirements-demo.txt`.
- **Data source change (Phase 1 follow-up):** UKP is behind an access form, so the default dataset is now **VAST**
  (open on GitHub; `scripts/download_data.py`). Dev and test samples come from VAST's own dev and test files, balanced over
  the three labels. Finding while building the loader: VAST's synthetic-neutral rows keep the comment's own topic in
  `new_topic` and put the paired topic in `topic_str`; reading `new_topic` would have produced thousands of contradictory
  labels, so the loader uses `topic_str` for them. About 94% of Neutral examples are synthetic, so the reports show accuracy by
  kind of example and warn that macro-F1 is flattered. The VAST download was tested for real from GitHub; no model scores exist yet.
- **Tests:** 208 in total, no Ollama and no internet needed (the demo is exercised with Streamlit's own test runner and
  the download script against a local web server). The full workflow was also run end to end against a fake Ollama server.
- **Still open:** a real run on your machine (all quality numbers), annotated transcripts, and timings on your GPU.
