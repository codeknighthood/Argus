# ARGUS: install and run, start to finish

Do the steps in order. Each step says what you should see. If something differs, jump to **Troubleshooting** at the bottom.
Commands are for Windows PowerShell; macOS and Linux notes are in brackets. All commands are run from the ARGUS folder
(the one that contains `app.py`).

Everything runs on your own computer. There are no accounts, no API keys and nothing to pay for.

## Part A. Set up (about 20 minutes, mostly downloading)

**Step 1. Install two programs** (skip any you already have)
- Python 3.10 or newer: https://www.python.org/downloads/ . On the first installer screen tick **Add python.exe to PATH**.
- Ollama: https://ollama.com . After installing, it runs in the background (llama icon near the clock).

**Step 2. Make a Python environment and install ARGUS's packages**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-demo.txt
```
[macOS/Linux: `source .venv/bin/activate` instead of the second line.]
You should see `(.venv)` at the start of the prompt. Do the `activate` line again every time you open a new window.
If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.

**Step 3. Download the language model** (4.7 GB, once)
```
ollama pull qwen2.5:7b
```

**Step 4. Check everything**
```
python scripts/check_setup.py --live
```
You want every line to say `[ OK ]` (two `[WARN]` lines about data are fine now: Step 5 fixes them). For anything marked
`[FAIL]` the script prints the exact command that fixes it. `--live` also asks the model one real question.

## Part B. Get the data and run ARGUS (about 30 to 60 minutes the first time)

**Step 5. Get the dataset** (no sign-up, about 11 MB)
```
python scripts/download_data.py
python scripts/prepare_data.py
```
This downloads **VAST** (Allaway and McKeown, 2020: comments from New York Times debates, each labelled pro, con or
neutral toward a topic) and builds a 450-example **dev** sample and a 900-example **test** sample, balanced across
Favor, Against and Neutral. If the download is blocked, the script prints two links: save each file into `data\raw\vast\`
under the same name and run `prepare_data.py` again.

> **Shortcut for Steps 4 to 7:** `python scripts/quickstart.py` runs check, download, prepare, a 60-sentence strategy
> comparison and the pipeline on the sample transcript, stopping at the first problem. Run the steps below one by one if
> you prefer to see each result.

**Step 6. A first look at stance detection** (60 sentences, a few minutes)
```
python scripts/evaluate.py --limit 60
```
Open `results\evaluation\summary.md`. It compares three prompting strategies (zero-shot, few-shot, chain-of-thought) with
confidence intervals. With only 60 sentences the intervals are wide: treat it as a smoke test, not a result.

**Step 7. The full pipeline on a transcript**
```
python scripts/run_pipeline.py
```
This analyses `data\sample\sample_transcript.txt` (an invented debate) and writes `results\sample_transcript.csv`
(open in Excel), `.json` and `.md`. For each sentence you get: argument or not, Claim or Premise, stance toward the
topic with a reason, and a Support or Attack link to the argument it answers. Your own debate:
```
python scripts/run_pipeline.py --transcript my_debate.txt --target "the minimum wage"
```
Format: one turn per line, `SPEAKER NAME: what they said`.

**Step 8. The demo**
```
streamlit run app.py
```
A browser tab opens. Click **Analyse debate**. The tabs show the **condensed view** (only the argumentative sentences,
grouped by stance, repeated points merged), an **argument map** (green arrows support, red attack), the full table and
downloads. Press Ctrl+C in the window to stop it.

## Part C. The evaluation you report (the synopsis section 5.8 work)

**Step 9. Choose a stance strategy on dev, then score test once**
```
python scripts/evaluate.py --consistency-runs 3
```
Read `summary.md`: pick the strategy with the best macro-F1 *that is not beaten significantly by another* (section 1 says).
Set it in `config.yaml` as `stance_strategy:` under `pipeline:`. Then, **once**:
```
python scripts/evaluate.py --split test --strategies <your choice>
```
These are the numbers for the report. Do not tune anything after seeing them.
How to read the report: Part D below and `docs/ERROR_ANALYSIS.md`.

**Step 10. Score argument extraction and relations on a transcript you annotate** (the dataset has no claim/premise labels)
```
python scripts/make_annotation_sheet.py
```
Fill `gold_component` in `data\annotation\components.csv` and `gold_relation` in `data\annotation\pairs.csv` in Excel (Save as
CSV) using the rules in `docs/ANNOTATION_GUIDE.md`. Ideally two team members annotate separately; compare with
`python scripts/annotator_agreement.py A.csv B.csv`. Then score the whole pipeline:
```
python scripts/evaluate_pipeline.py --target "the minimum wage"
```
It reports Claim/Premise/None and Support/Attack/None scores and, for every link the pipeline missed, why.
You can also score each stage alone: `run_annotated_eval.py` (extraction) and `run_relation_eval.py` (relations).

**Step 11. Find out why it is wrong**
Open `results\evaluation\errors_<run>_test.csv`, fill the `error_type` column (1 to 7, see `docs/ERROR_ANALYSIS.md`) for the
first 50 to 100 rows, then:
```
python scripts/tally_errors.py results\evaluation\errors_<run>_test.csv
```

## Part D. Reading the results honestly

- **Neutral is easy in this dataset.** About 94% of VAST's Neutral examples pair a comment with an unrelated topic. The
  report has a table of accuracy by kind of example; judge the model mainly on Favor against Against and on the
  non-synthetic examples, and say so in the report.
- A difference between strategies counts only if the report's significance line says so. Small test samples cannot
  separate strategies that are a few points apart.
- Use dev to choose and test once to report. State the sample sizes next to every number.
- The model, the prompts and the data are all stated in `results\evaluation\summary.md` and `summary.json`.
  Keep those files with your report.

## Troubleshooting

| You see | Do this |
|---|---|
| `python` is not recognised | Reinstall Python and tick "Add python.exe to PATH", then open a new PowerShell window. |
| `No module named ...` | The environment is not active. Run `.venv\Scripts\activate`, then `pip install -r requirements.txt -r requirements-demo.txt`. |
| `Cannot reach Ollama` | Start Ollama (search "Ollama" in the Start menu), wait ten seconds, run again. |
| `Model 'qwen2.5:7b' is not downloaded` | `ollama pull qwen2.5:7b` |
| Very slow, GPU idle | Run `ollama ps` while a script runs; the PROCESSOR column should say GPU. Update the NVIDIA driver; close games and browsers using the GPU. |
| `... not found. The data has not been prepared yet` | Do Step 5. |
| Download blocked or fails | Follow the two links the script prints and save the files by hand (Step 5). |
| Excel mangles the CSV | Open Excel first, then Data > From Text/CSV, encoding UTF-8. Save as CSV (comma delimited). |
| "Fix these cells in the sheet" | The message lists each bad cell with its row. Valid labels are in `docs/ANNOTATION_GUIDE.md`. |
| `streamlit` is not recognised | `pip install -r requirements-demo.txt` with the environment active. |
| A long run was interrupted | Run the same command again: answers are cached, so only the missing work is redone. |
| Want to try another model | `ollama pull llama3.1:8b`, then `python scripts/evaluate.py --models qwen2.5:7b llama3.1:8b`. |
