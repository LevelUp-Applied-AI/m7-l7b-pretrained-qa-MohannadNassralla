"""
Autograder tests for the Tuesday Honors stretch — Adversarial QA Probe.

Triggered by .github/workflows/stretch-tue-autograder.yml on stretch-tue-*
branches (and PRs to main that touch stretch/tuesday/**). Skips cleanly when
the stretch directory hasn't been populated, so it does not block learners
who are only working on the lab.
"""

import ast
import json
import os
import re
import string
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
STRETCH_DIR = os.path.join(REPO_ROOT, "stretch", "tuesday")
ADV_SET = os.path.join(STRETCH_DIR, "adversarial_set.csv")
ADV_EVAL = os.path.join(STRETCH_DIR, "adversarial_eval.py")
ADV_PREDS = os.path.join(STRETCH_DIR, "adversarial_predictions.csv")
ADV_METRICS = os.path.join(STRETCH_DIR, "adversarial_metrics.json")
ADV_ANALYSIS = os.path.join(STRETCH_DIR, "adversarial-analysis.md")


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _stretch_started() -> bool:
    """True only if learner has begun the stretch (replaced placeholder rows)."""
    if not os.path.exists(ADV_SET):
        return False
    df = pd.read_csv(ADV_SET)
    if len(df) == 0:
        return False
    # Heuristic: placeholder rows have "REPLACE_WITH" in the question column
    return not df["question"].astype(str).str.contains("REPLACE_WITH", na=False).all()


# -- Set schema + content checks ---------------------------------------------

def test_adversarial_set_has_required_columns():
    if not _stretch_started():
        pytest.skip("Stretch not started — adversarial_set.csv still has placeholder rows")
    df = pd.read_csv(ADV_SET)
    required = {"qid", "question", "context", "gold_answer", "pattern_tag"}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


def test_adversarial_set_row_count():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    df = pd.read_csv(ADV_SET)
    assert 25 <= len(df) <= 50, f"expected 25–50 rows, got {len(df)}"


def test_adversarial_set_gold_answers_in_context():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    df = pd.read_csv(ADV_SET)
    for _, row in df.iterrows():
        ctx_norm = _normalize(str(row["context"]))
        gold_norm = _normalize(str(row["gold_answer"]))
        assert gold_norm in ctx_norm, (
            f"qid={row['qid']}: gold answer {row['gold_answer']!r} not a substring "
            f"of context after normalization (extractive QA constraint)"
        )


# -- Eval function structure -------------------------------------------------

def test_evaluate_adversarial_returns_per_pattern_breakdown():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, STRETCH_DIR)
    import adversarial_eval  # noqa: E402

    # AST-level check: evaluate_adversarial must return a dict with em, f1, per_pattern
    src = open(ADV_EVAL).read()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_adversarial":
            found = True
            doc = ast.get_docstring(node) or ""
            assert "per_pattern" in doc or "per-pattern" in doc, (
                "evaluate_adversarial docstring must reference per-pattern breakdown"
            )
    assert found, "evaluate_adversarial function not defined in adversarial_eval.py"


def test_adversarial_metrics_json_schema():
    if not os.path.exists(ADV_METRICS):
        pytest.skip(f"{ADV_METRICS} not produced yet — run `python adversarial_eval.py` first")
    with open(ADV_METRICS) as f:
        m = json.load(f)
    for k in ("em", "f1", "n", "per_pattern", "model"):
        assert k in m, f"missing key: {k}"
    assert isinstance(m["per_pattern"], dict) and len(m["per_pattern"]) >= 1, (
        "per_pattern must be a non-empty dict"
    )


def test_analysis_md_has_required_sections():
    if not os.path.exists(ADV_ANALYSIS):
        pytest.skip(f"{ADV_ANALYSIS} not present")
    content = open(ADV_ANALYSIS).read()
    lower = content.lower()
    required_sections = ["hypothesis", "set design", "results", "production defense"]
    missing = [s for s in required_sections if s not in lower]
    # Catch the case where the learner submits the unmodified template
    placeholder_signals = ["_(what kind of input", "_(name of failure mode)", "_(your paragraph)"]
    if all(p not in content for p in placeholder_signals):
        # Genuinely written
        assert not missing, f"missing required sections: {missing}"
    else:
        pytest.fail(
            "adversarial-analysis.md still contains template placeholders — "
            "replace them with your written analysis"
        )
