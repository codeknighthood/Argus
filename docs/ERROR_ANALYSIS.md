# Error analysis guide (Phase 6, synopsis 5.8)

A score tells you how often the model is wrong. Error analysis tells you **why**, which is what the report needs.
The program can show where mistakes cluster; only you can say what caused them.

## Procedure

1. Run the comparison on the dev split and open `results/evaluation/summary.md`.
   ```
   python scripts/evaluate.py
   ```
2. Pick the best strategy and set it as `stance_strategy` in `config.yaml`.
3. Run the **test** split once for the numbers you will report. Do not choose a strategy from test scores.
   ```
   python scripts/evaluate.py --split test --strategies few_shot
   ```
4. Open `results/evaluation/errors_<run>_test.csv` in Excel. The mistakes are already in random order, so tagging the
   first 50 to 100 rows gives a fair sample of all mistakes. Read each sentence, its gold label and the model's answer
   and explanation, then fill `error_type` (a number or a name, see below). Use `notes` for anything you want to quote.
5. Count the causes:
   ```
   python scripts/tally_errors.py results/evaluation/errors_<run>_test.csv --out results/evaluation/causes.md
   ```
6. If two people tag the same file independently, report their agreement before discussing:
   ```
   python scripts/annotator_agreement.py errorsA.csv errorsB.csv --column error_type
   ```

## The seven causes

Tag the **main** reason the model's answer differs from the gold label. When two fit, pick the one that, if fixed,
would have changed the answer.

| # | Cause | Use it when |
|---|---|---|
| 1 | implicit argument | The stance is only implied. Getting it right needs an inference the sentence does not state. |
| 2 | ambiguous statement | Reasonable readers could disagree about the stance. You would not be sure yourself. |
| 3 | missing context | The sentence depends on what came before (it, this, that, "I agree"), which the model did not see or did not use. |
| 4 | label noise | The gold label looks wrong to you and the model's answer is defensible. Be strict: only when you are confident. |
| 5 | negation or sarcasm | A negation, irony or sarcasm reverses the surface meaning and the model took it literally. |
| 6 | wrong target | The sentence takes a side on something else, such as a related policy, a person or a source, not on the policy under debate. |
| 7 | other | None of the above. If this is more than about 15% of your tags, the categories need revising; say so in the report. |

## Reading the automatic cues

`summary.md` and the `error_kind` / `auto_tags` columns are computed, not judged.

- **Shape of the mistake.** *Polarity flip* means Favor called Against or the reverse. *Over-reading* means a side was
  found in a sentence the gold label calls Neutral. *Under-reading* means a real stance was called Neutral. Most
  corpus labelling noise sits at the Neutral boundary, so over- and under-reading are the places to look first.
- **Sentence features** (negation, hedge, question, pronoun start ...) show where errors cluster. A feature with a high
  error rate is a **lead**: go and read those mistakes. It does not prove the feature causes the errors, because the
  feature may simply travel with something else, and with few sentences a gap can be luck.
- **Mistakes every strategy makes.** Section 6 of the report counts sentences all runs get wrong. These point at label
  noise or at something no prompt fixes, so read them first.

## What you can and cannot claim

- You can report each score with its 95% interval. Say "A scored higher than B" only if the McNemar p-value is
  below 0.05; otherwise say the difference is within noise.
- You can report how often each cause appeared **among the mistakes you tagged**. Do not call it the share of all
  mistakes unless you tagged all of them or a random sample (the sheet is already shuffled).
- You cannot say the results hold for other models, other corpora or real debate transcripts. The corpus is
  one dataset of crowd-labelled web sentences, and the model is one 7B model.
- Treat "label noise" tags as your judgement. If many mistakes are tagged label noise, say that the ceiling on this
  corpus is below 100% and report agreement between you and a second tagger.

## Suggested structure for the report

1. Setup: model, prompts, split sizes, how the strategy was chosen (dev) and scored (test).
2. Results table with intervals, and the significance tests between strategies.
3. Per-label and per-topic scores; the weakest label and topic, with a sentence on why you think so.
4. Error causes (table from `tally_errors.py`) with two or three quoted examples per main cause.
5. Pipeline results on the annotated transcript (`scripts/evaluate_pipeline.py`), including why links were missed.
6. Limits: sample sizes, annotator agreement, one model, one corpus.
