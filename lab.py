"""
Module 7 Week B — Applied Lab: Pre-Trained QA Evaluation on Tech/Entertainment News.

Implement the functions below. See the lab guide for full task descriptions.
"""

import json
import os
import re
import string
from collections import Counter

import pandas as pd
from transformers import pipeline


# -- Helpers (provided — do NOT modify) --------------------------------------

def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI smoke fixture), otherwise the default
    path to the curated tech/entertainment QA CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/tech_news_qa.csv")


def get_qa_model_name() -> str:
    """Return the QA model name. Provided helper. Do not modify."""
    return "distilbert-base-cased-distilled-squad"


def load_examples(data_path: str) -> pd.DataFrame:
    """
    Load the curated QA CSV.

    Returns a DataFrame with columns: qid, question, context, gold_answer.

    The default `data/tech_news_qa.csv` ships ~1,000 extractive QA tuples
    derived from glnmario/news-qa-summarization (CNN tech / entertainment
    slice). Each gold answer is a literal substring of its context, by
    construction (filtered during curation).

    Provided helper. Do not modify.
    """
    df = pd.read_csv(data_path)
    required = {"qid", "question", "context", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    return df


# -- Task 1: Normalization + EM + F1 (same as drill) -------------------------

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization (see drill / reading)."""
    # 1. Lowercase
    s = str(s).lower()
    
    # 2. Strip standalone articles (a, an, the) using word-boundary regex
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    
    # 3. Strip all punctuation characters
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    
    # 4. Collapse whitespace and strip leading/trailing spaces
    s = " ".join(s.split())
    
    return s


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalized prediction equals normalized gold."""
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0


def token_f1(pred: str, gold: str) -> float:
    """
    Token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty -> 1.0
      - one empty -> 0.0
    Returns float in [0.0, 1.0]; never NaN.
    """
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    
    # Handle empty cases explicitly to avoid division by zero (NaN)
    if not norm_pred and not norm_gold:
        return 1.0
    if not norm_pred or not norm_gold:
        return 0.0
        
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    
    # Compute multiset overlap via Counter collections
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    
    # If no tokens overlap, F1 is explicitly 0.0
    if overlap == 0:
        return 0.0
        
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


# -- Task 2: Build the QA pipeline -------------------------------------------

def build_qa_pipeline(model_name: str):
    """Construct a Hugging Face question-answering pipeline."""
    return pipeline("question-answering", model=model_name)


# -- Task 3: Predict one answer ---------------------------------------------

def predict_one(qa, question: str, context: str) -> str:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the answer STRING only (not the full pipeline output dict).
    """
    result = qa(question=question, context=context)
    return result.get("answer", "")


# -- Task 4: Evaluate over the dataset ---------------------------------------

def evaluate_qa(qa, examples: pd.DataFrame) -> dict:
    """
    Evaluate the QA pipeline over a DataFrame of examples.

    Returns:
        {
          "em": float,   # mean EM
          "f1": float,   # mean token-F1
          "n": int,
          "predictions": [
            {qid, question, context_excerpt, gold_answer, predicted_answer, em, f1},
            ...
          ],
        }
    context_excerpt is the first 80 chars of the context (CSV-friendly).
    """
    predictions_list = []
    total_em = 0
    total_f1 = 0.0
    n = len(examples)
    
    for _, row in examples.iterrows():
        qid = str(row["qid"])
        question = str(row["question"])
        context = str(row["context"])
        gold_answer = str(row["gold_answer"])
        
        # Call Task 3 thin wrapper to isolate answer string
        predicted_answer = predict_one(qa, question, context)
        
        # Calculate metric evaluation scores
        em_score = exact_match(predicted_answer, gold_answer)
        f1_score = token_f1(predicted_answer, gold_answer)
        
        total_em += em_score
        total_f1 += f1_score
        
        # Generate the safe 80 character excerpt
        context_excerpt = context[:80]
        
        predictions_list.append({
            "qid": qid,
            "question": question,
            "context_excerpt": context_excerpt,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "em": em_score,
            "f1": f1_score
        })
        
    # Calculate macro aggregate means across the subset slice
    mean_em = total_em / n if n > 0 else 0.0
    mean_f1 = total_f1 / n if n > 0 else 0.0
    
    return {
        "em": mean_em,
        "f1": mean_f1,
        "n": n,
        "predictions": predictions_list
    }


# -- Task 5: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    examples = load_examples(get_data_path())
    qa = build_qa_pipeline(get_qa_model_name())
    result = evaluate_qa(qa, examples)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("qa_predictions.csv", index=False)

    # Write metrics JSON
    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "model": get_qa_model_name(),
    }
    with open("qa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()