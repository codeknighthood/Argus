# ARGUS

**An LLM-Based Framework for Argument Mining and Stance Detection in Policy Debates**

B.Tech CSE Project-1, Medicaps University, Indore (Aug–Dec 2026)
Riya Gandhi · Rohan R. Joshi · Sakshi Joshi — Guide: Prof. Shivali Shukla

ARGUS reads a debate transcript (text or text-based PDF) and returns, for every sentence:

- its role: **Claim**, **Premise** or **None**, with a standalone restatement of the argument
- its **stance** (Favor / Against / Neutral) toward a policy you name, with a short explanation
- a **Support / Attack** link to the nearest earlier argument it answers

It runs fully offline on a local model (Qwen2.5-7B through [Ollama](https://ollama.com)). No API keys, no accounts, no paid services. Every model reply is constrained to a JSON schema, validated, retried on failure and cached on disk, so runs are reproducible.

## Features

- Sentence segmentation with speaker and two sentences of context
- Three stance prompting strategies: zero-shot, few-shot, chain-of-thought
- Argument extraction and nearest-first relation analysis
- Output as JSON, CSV and Markdown
- Streamlit demo: condensed view grouped by stance, argument map, PDF upload
- Evaluation tools: accuracy, precision, recall, F1, bootstrap confidence intervals, exact McNemar test, Cohen's kappa, error-cause tagging
- 208 unit tests that need neither the model nor the internet

## Results

Dataset: [VAST](https://github.com/emilyallaway/zero-shot-stance) (Allaway and McKeown, EMNLP 2020), balanced samples of 450 (development) and 900 (test) sentences. Model: Qwen2.5-7B, temperature 0.

**Stance detection (test sample, n = 900, 0 failed calls)**

| Strategy | Accuracy (95% CI) | Macro-F1 |
|---|---|---|
| **Zero-shot** | **62.7%** (59.3–65.8) | **62.4%** |
| Chain-of-thought | 62.0% (58.9–65.2) | 61.8% |
| Few-shot | 58.7% (55.6–61.9) | 57.7% |

Zero-shot was chosen on the development sample (macro-F1 65.2%). It was significantly better than few-shot (McNemar p = 0.009 dev, 0.003 test) and not significantly different from chain-of-thought (p = 0.592 dev, 0.728 test).

**Argument mining on an annotated sample transcript (67 sentences, 131 pairs)**

| Task | Macro-F1 | Cohen's kappa (model vs annotation) |
|---|---|---|
| Claim / Premise / None | 0.841 | 0.753 |
| Support / Attack / None | 0.556 | 0.494 |

The relation score rests on a single Attack example and should be read as indicative. Full tables are in [`docs/results/`](docs/results/).

**Things to know before quoting these numbers**

- About 94% of VAST's Neutral examples are synthetic (a comment paired with an unrelated topic). For zero-shot these were the hardest examples (56.9%); on the other examples accuracy was about 65%.
- The model recognises Against well (recall 85%) but Favor poorly (recall 49%).
- In a hand-tagged sample of 60 mistakes, the main causes were wrong target (42%) and implicit arguments (32%).
- One model, one dataset, one small annotated transcript: the results do not claim to hold for other models or real debates.

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com), about 5 GB of disk for the model. A GPU helps but is not required.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
ollama pull qwen2.5:7b

python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py      # should end
