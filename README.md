# Module 7 Week B — Applied Lab: Pre-Trained QA Evaluation on Tech/Entertainment News

This is the starter repo for the Module 7 Week B Applied Lab.

The full lab guide is at <a href="https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-7/16a827b1" target="_blank">the lab guide page</a> — read it first.

## Quick start

```bash
pip install -r requirements.txt
git checkout -b lab-7b-pretrained-qa
# Implement the TODOs in lab.py
python lab.py    # runs full evaluation; first run downloads ~250 MB
```

The first call to `pipeline("question-answering", ...)` downloads the model. Plan ~3 minutes for the first run; subsequent runs use the cached weights.

## What you will produce

Committed:
- `lab.py` — your implementation
- `qa_predictions.csv` — ~1,000 rows with per-question EM and F1
- `qa_metrics.json` — aggregate metrics
- `qa-evaluation-report.md` — one-page report with three failure modes

**No model file** — pre-trained models load from Hugging Face Hub at runtime.

## Local testing

The starter includes a smoke test you can run before pushing:

```bash
DATA_PATH=fixtures/tiny_qa.csv python lab.py
pytest tests/ -v
```

## Submission

Open a Pull Request from your `lab-7b-pretrained-qa` branch into `main`. The autograder runs automatically on every push. PR description requirements are in the lab guide.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
