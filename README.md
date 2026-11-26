# ARGUS

An LLM-Based Framework for Argument Mining and Stance Detection in Policy Debates.
Project-1, B.Tech CSE, Medicaps University (Aug-Dec 2026).

ARGUS runs fully offline on a local model through Ollama: no API keys, no accounts, no rate limits.

> **Fresh machine?** Follow `INSTALL.md`: one guide, in order, from installing Python to the final evaluation.

## Setup (Phase 0)

1. Install Ollama from https://ollama.com and start it (it runs in the background / system tray).
2. Download the model (about 4.7 GB, one time):

```bash
ollama pull qwen2.5:7b
```

3. Set up Python:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

4. Run the check:

```bash
python scripts/smoke_test.py
```

You should see a stance and a short reason for each sample statement, then `Phase 0 OK`.
Run it a second time: it should say `(from cache)` and return instantly.

## Getting the data (Phase 1)

ARGUS uses **VAST** (Allaway and McKeown, 2020, "Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic
Representations", EMNLP): comments from New York Times debates, each paired with a topic and labelled pro, con or neutral
toward it (Favor / Against / Neutral here). It is published openly on GitHub, so no registration is needed, and it
matches the synopsis's cross-target stance setting (section 2.3). The older UKP corpus needs an access form and is no longer
used (the loader is still in the code as an option).

```bash
python scripts/download_data.py      # about 11 MB, no sign-up
python scripts/inspect_data.py       # optional: look at what you have
python scripts/prepare_data.py       # dev sample (450) and test sample (900), balanced over the three labels
```

This writes `data/processed/all.jsonl`, `dev.jsonl` (iterate on prompts here) and `test.jsonl` (report final numbers here,
once per method). Note: most VAST "Neutral" examples are synthetic (a comment paired with an unrelated topic), so Neutral
is easier than the other labels; the evaluation report shows accuracy by kind of example. If the download is blocked, the
script prints the links to save the two files by hand. Full steps: `INSTALL.md`.

Try the transcript segmenter (speaker + context per sentence):

```bash
python scripts/segment_demo.py
```

Run the tests (synthetic data, no download needed):

```bash
python -m unittest discover -s tests -t . -v
```

## Stance detection (Phase 2)

Score the three prompting strategies on the dev sample. Start small to check it works:

```bash
python scripts/run_stance.py --strategy zero_shot --limit 60
python scripts/run_stance.py --strategy zero_shot
python scripts/run_stance.py --strategy few_shot
python scripts/run_stance.py --strategy cot
```

Each run prints accuracy, per-class precision/recall/F1 and a confusion matrix, and saves
`results/stance_<strategy>_dev.jsonl` (every prediction with its explanation) and `.json` (the scores).
Answers are cached, so you can stop and re-run safely. Tune prompts on `dev` only; use `--split test`
once per strategy for the numbers you report.

## Argument extraction (Phase 3)

Run extraction on the dev sample (VAST has no argument / non-argument labels, so this shows how roles line up with stance rather than a score):

```bash
python scripts/run_extraction.py --limit 60
python scripts/run_extraction.py
```

The dataset has no claim/premise labels, so to score Claim / Premise / None you annotate a transcript yourself.
Rules are in `docs/ANNOTATION_GUIDE.md`.

```bash
python scripts/make_annotation_sheet.py --kind components   # writes data/annotation/components.csv
# fill the gold_component column in Excel (Claim / Premise / None), save as CSV
python scripts/run_annotated_eval.py --target "the minimum wage"
python scripts/annotator_agreement.py A.csv B.csv           # two annotators: Cohen's kappa
```

`data/sample/` holds three invented debates for trying the tools and the demo: `sample_transcript.txt`, and two PDFs laid out like real transcripts (`council_hearing_congestion_charge.pdf`, `tv_panel_compulsory_voting.pdf`). The demo (`streamlit run app.py`) accepts `.txt` and text-based `.pdf` transcripts; headers, footers, page numbers, time codes and notes like [Applause] are removed first (`argus/pdf_text.py`). Scanned PDFs have no text layer and are not supported.

`data/sample/sample_transcript.txt` is a short invented debate for trying the tools. Use a real
transcript (one turn per line, `SPEAKER NAME: what they said`) with `--transcript` for the real evaluation.

## Argument relations (Phase 4)

Does a later argument support or attack an earlier one? Like claim/premise labels, this needs your own annotation
(the rules are in `docs/ANNOTATION_GUIDE.md`):

```bash
python scripts/make_annotation_sheet.py --kind pairs        # writes data/annotation/pairs.csv
# fill the gold_relation column in Excel (Support / Attack / None), save as CSV
python scripts/run_relation_eval.py --target "the minimum wage"
python scripts/annotator_agreement.py A_pairs.csv B_pairs.csv --column gold_relation
```

`make_annotation_sheet.py` with no `--kind` writes both sheets. It never overwrites a sheet that already exists
unless you pass `--force`, so your annotations are safe; use `--kind pairs` to add the pairs sheet next to a
components sheet you have already filled in.

The full phase-by-phase log, with what is verified and what is still open, is in `docs/PHASES.md`.

## The full pipeline (Phase 5)

Transcript in, structured output out (synopsis 5.7): for every sentence, its role (Claim / Premise / none), a
standalone restatement of the argument, the stance toward the policy with an explanation, and a Support / Attack link
to the nearest earlier argument it responds to.

```bash
python scripts/run_pipeline.py                                   # the sample transcript
python scripts/run_pipeline.py --limit-segments 8                # quick try on the first 8 sentences
python scripts/run_pipeline.py --transcript my_debate.txt --target "the minimum wage"
```

Transcript format: one turn per line, `SPEAKER NAME: what they said`. Writes `results/<name>.json` (with the settings
used), `.csv` (opens in Excel) and `.md` (a compact table). Each sentence takes several model calls, so a long
transcript takes a while; answers are cached, so you can stop and re-run safely.

Defaults (stance strategy, context window, relation window, ...) are in the `pipeline:` section of `config.yaml`;
command-line options override them. If one model call fails, that row records the problem in its `error` column and
the run carries on.

## Evaluation and error analysis (Phase 6)

Compare the three stance prompting strategies on the same sentences, with confidence intervals and significance tests,
then find out why the model is wrong (synopsis 5.8).

```bash
python scripts/evaluate.py --limit 60                         # quick look on 60 dev sentences
python scripts/evaluate.py                                    # all three strategies, whole dev sample
python scripts/evaluate.py --consistency-runs 3               # also: does the answer change between runs?
python scripts/evaluate.py --split test --strategies few_shot # the numbers to report: run on test ONCE
```

Choose the strategy on **dev**, set it as `stance_strategy` in `config.yaml`, and score **test** once. Reports go to
`results/evaluation/`: `summary.md` (read this first), `summary.json`, every prediction, and `errors_<run>_<split>.csv`,
the mistakes in random order with an empty `error_type` column.

```bash
# tag error_type in the errors CSV in Excel (1 to 7, see docs/ERROR_ANALYSIS.md), then:
python scripts/tally_errors.py results/evaluation/errors_<run>_test.csv

# score the whole pipeline on a transcript you annotated in Phases 3 and 4
python scripts/evaluate_pipeline.py --target "the minimum wage"
```

`evaluate_pipeline.py` runs segmentation, extraction and relations together and scores them against your sheets,
and for every missed link says why (argument not detected, outside the window, linked to a nearer argument ...).
The sheets must come from the same transcript and settings. Stance is skipped there (a transcript has no stance
labels); `pipeline.stance_on: none` in `config.yaml` does the same for `run_pipeline.py`.
The full procedure and how to read the report: `docs/ERROR_ANALYSIS.md`.

## Demo and setup tools (Phase 7)

```bash
python scripts/check_setup.py --live   # is this computer ready? prints the exact fix for anything that is not
python scripts/quickstart.py           # check, download, prepare, trial evaluation and pipeline run, in order
streamlit run app.py                   # the demo: paste a transcript, see the condensed view and argument map
```

The demo's **condensed view** keeps only the sentences that carry an argument, groups them by stance and merges repeated
points, so a long debate can be read quickly; the **argument map** draws Support and Attack links. Everything is computed
by the same pipeline as `run_pipeline.py`. The step-by-step guide for a fresh machine is `INSTALL.md`.

## Switching models

Edit `model:` in `config.yaml`, for example `llama3.1:8b` (after `ollama pull llama3.1:8b`).
Comparing two models is useful for the evaluation phase.

## Layout

| Path | Purpose |
|---|---|
| `argus/schema.py` | Output schema from synopsis 5.7 (labels: Favor / Against / Neutral) |
| `argus/llm.py` | Ollama client: schema-constrained JSON, validation, retries, disk cache |
| `argus/data.py` | Common `Record`, VAST (and optional UKP) loader, cleaning, de-duplication, dev/test sampling |
| `argus/segment.py` | Transcript to sentence segments with speaker and carried-over context |
| `argus/stance.py` | Zero-shot, few-shot and chain-of-thought stance detection |
| `argus/extract.py` | Claim / Premise / None, plus a self-contained restatement of the argument |
| `argus/relations.py` | Support / Attack / None between two arguments; nearest-first linking in a transcript |
| `argus/pipeline.py` | The full pipeline: transcript to one structured record per sentence |
| `argus/output.py` | JSON, CSV and table output in the synopsis 5.7 format, plus summary counts |
| `argus/annotation.py` | Annotation sheets and annotator agreement for claim / premise and relation labels |
| `argus/metrics.py` | Accuracy, precision, recall, F1, confusion matrix, Cohen's kappa, bootstrap interval, McNemar test |
| `argus/compare.py`, `argus/report.py` | Strategy / model comparison and the markdown report |
| `argus/analysis.py` | Error shapes, sentence-feature cues, the seven error causes |
| `argus/pipeline_eval.py` | Score the whole pipeline on an annotated transcript |
| `argus/demo.py`, `app.py` | Condensed view, argument map and the Streamlit demo |
| `argus/setup_check.py` | The checks behind `scripts/check_setup.py` |
| `argus/evaluation.py` | Run a stage over a dataset, per-topic scores, result files |
| `argus/prompting.py`, `argus/prompts/` | Versioned prompt files, so experiments are reproducible |
| `tests/` | Unit tests (no Ollama needed); `tests/fake_ollama_server.py` is a dev-only fake server |
| `INSTALL.md` | Step-by-step install and run guide |
| `docs/PHASES.md` | Phase-by-phase development log |
| `docs/ERROR_ANALYSIS.md` | How to tag and report the causes of mistakes |
| `scripts/` | Runnable scripts for each phase |
| `data/raw/` | Datasets (not committed) |
| `results/` | Evaluation outputs (not committed) |

## Phase status

- [x] Phase 0: setup and decisions
- [x] Phase 1: data and preprocessing (code done; needs the real corpus downloaded)
- [x] Phase 2: stance detection (code done; needs a real run on your machine)
- [x] Phase 3: argument extraction (code done; needs a real run and an annotated transcript)
- [x] Phase 4: argument relations (code done; needs a real run and an annotated pairs sheet)
- [x] Phase 5: integrated pipeline (code done; needs a real run)
- [x] Phase 6: evaluation and error analysis (code done; needs real runs and your error tagging)
- [x] Phase 7: demo, setup checker, install guide (code done; needs a run on your machine)
#   A r g u s  
 