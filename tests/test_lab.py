"""
Autograder tests for Module 7 Week B — Applied Lab.

CI runs `python lab.py` with DATA_PATH=fixtures/tiny_qa.csv first (produces
qa_predictions.csv and qa_metrics.json against the smoke fixture); these tests
then verify both the function-level correctness and the produced artifacts.
"""

import json
import math
import os
import sys

import pandas as pd
import pytest

# Add repo root to sys.path so we can import the learner's lab.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lab  # noqa: E402


# -- Function-level correctness ---------------------------------------------

def test_normalize_answer_lowercases_strips_articles_punct():
    assert lab.normalize_answer("The Eiffel Tower") == "eiffel tower"


def test_normalize_answer_strips_articles_with_word_boundaries():
    out = lab.normalize_answer("Thereby a fact")
    assert "thereby" in out
    assert " a " not in f" {out} "


def test_exact_match_known_cases():
    assert lab.exact_match("The Eiffel Tower!", "eiffel tower") == 1
    assert lab.exact_match("Paris", "London") == 0


def test_token_f1_perfect_overlap():
    assert lab.token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap():
    assert lab.token_f1("apple banana", "carrot date") == 0.0


def test_token_f1_partial_overlap_known_value():
    # "cat sat mat" (3 tokens) vs "cat sat on mat" (4 tokens) after normalization
    # overlap = 3, P=1.0, R=0.75, F1 = 6/7
    f1 = lab.token_f1("cat sat mat", "cat sat on mat")
    assert math.isclose(f1, 6 / 7, rel_tol=1e-6)


def test_token_f1_handles_empty_prediction():
    out = lab.token_f1("", "the answer")
    assert out == 0.0
    assert not math.isnan(out)


# -- Pipeline construction ---------------------------------------------------

@pytest.fixture(scope="module")
def qa_pipeline():
    return lab.build_qa_pipeline(lab.get_qa_model_name())


def test_build_qa_pipeline_returns_callable(qa_pipeline):
    assert callable(qa_pipeline)


def test_predict_one_returns_string(qa_pipeline):
    out = lab.predict_one(
        qa_pipeline,
        question="Where was Alan Turing born?",
        context="Alan Turing was born in Maida Vale, London.",
    )
    assert isinstance(out, str)
    assert len(out) > 0


# -- Evaluation harness ------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_examples():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tiny_qa.csv")
    return pd.read_csv(fixture_path)


def test_evaluate_qa_returns_required_keys(qa_pipeline, fixture_examples):
    result = lab.evaluate_qa(qa_pipeline, fixture_examples)
    assert isinstance(result, dict)
    for k in ("em", "f1", "n", "predictions"):
        assert k in result, f"missing key: {k}"


def test_evaluate_qa_predictions_have_required_columns(qa_pipeline, fixture_examples):
    result = lab.evaluate_qa(qa_pipeline, fixture_examples)
    assert len(result["predictions"]) == len(fixture_examples)
    required = {"qid", "question", "context_excerpt", "gold_answer", "predicted_answer", "em", "f1"}
    for pred in result["predictions"]:
        assert required.issubset(pred.keys()), f"missing columns: {required - pred.keys()}"


def test_evaluate_qa_smoke_f1_threshold(qa_pipeline, fixture_examples):
    """On the tiny_qa fixture (gold answers are literal substrings of context),
    the model should achieve aggregate F1 >= 0.5 — a liveness check."""
    result = lab.evaluate_qa(qa_pipeline, fixture_examples)
    assert result["f1"] >= 0.5, f"smoke F1 {result['f1']} below threshold 0.5"


# -- Artifact tests (run after `python lab.py` produces them in CI) ---------

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
PREDICTIONS_CSV = os.path.join(REPO_ROOT, "qa_predictions.csv")
METRICS_JSON = os.path.join(REPO_ROOT, "qa_metrics.json")


def test_qa_predictions_csv_has_required_columns():
    if not os.path.exists(PREDICTIONS_CSV):
        pytest.skip("qa_predictions.csv not produced yet (run `python lab.py` first)")
    df = pd.read_csv(PREDICTIONS_CSV)
    required = {"qid", "question", "context_excerpt", "gold_answer", "predicted_answer", "em", "f1"}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


def test_qa_metrics_json_schema():
    if not os.path.exists(METRICS_JSON):
        pytest.skip("qa_metrics.json not produced yet (run `python lab.py` first)")
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    for k in ("em", "f1", "n", "model"):
        assert k in metrics, f"missing key: {k}"
    assert isinstance(metrics["em"], float)
    assert isinstance(metrics["f1"], float)
    assert isinstance(metrics["n"], int)
    assert isinstance(metrics["model"], str)


def test_main_produces_artifacts():
    """End-to-end: the artifacts exist after `python lab.py` runs in CI."""
    assert os.path.exists(PREDICTIONS_CSV), "qa_predictions.csv not found"
    assert os.path.exists(METRICS_JSON), "qa_metrics.json not found"


# -- Report-quality structural check ----------------------------------------

REPORT_MD = os.path.join(REPO_ROOT, "qa-evaluation-report.md")


def test_qa_evaluation_report_exists_and_nonempty():
    if not os.path.exists(REPORT_MD):
        pytest.skip("qa-evaluation-report.md not present yet — TA verifies report submission")
    with open(REPORT_MD) as f:
        content = f.read()
    assert len(content) >= 600, f"report length {len(content)} < 600 chars"

    # Loose check that report references three failure modes — looks for
    # numbered list markers or the word "failure" + a small number of distinct
    # patterns. This is a structural check, not a content audit.
    lower = content.lower()
    failure_keywords = ["failure", "mode", "pattern", "error"]
    keyword_hits = sum(1 for kw in failure_keywords if kw in lower)
    assert keyword_hits >= 2, "report doesn't appear to discuss failure modes"
