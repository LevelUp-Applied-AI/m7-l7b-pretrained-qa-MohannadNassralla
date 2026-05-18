"""
Learner-runnable smoke tests (small subset of the autograder).

Run these locally to sanity-check your implementation before pushing:

    pytest tests/ -v

These are NOT the autograder. The autograder lives in tests/ at the repo root
and runs in GitHub Actions on every push.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lab  # noqa: E402


def test_normalize_answer_basic():
    assert lab.normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_basic():
    assert lab.exact_match("The Eiffel Tower!", "eiffel tower") == 1
    assert lab.exact_match("Paris", "London") == 0


def test_token_f1_perfect_match():
    assert lab.token_f1("hello world", "hello world") == 1.0


def test_token_f1_empty_handling():
    out = lab.token_f1("", "an answer")
    assert out == 0.0
    assert not math.isnan(out)
