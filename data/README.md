# Data

`tech_news_qa.csv` (~1,000 extractive QA tuples) is the lab's evaluation set. It is a curated tech / entertainment / digital-culture slice of <a href="https://huggingface.co/datasets/glnmario/news-qa-summarization" target="_blank">glnmario/news-qa-summarization</a> (CNN news articles with crowd-sourced QA pairs and reference summaries). The curation script lives in the curriculum repo at `scripts/curation/curate_m7_news_qa.py`.

Columns: `qid, article_id, question, context, gold_answer`. Every gold answer is a literal substring of its context (extractive QA constraint, enforced during curation). `article_id` lets you join this CSV to the matching summarization corpus in Integration 7B's repo (`tech_news_articles.csv`) — useful for cross-task analysis.

The lab pipeline reads this file by default. CI smoke runs override the path via `DATA_PATH=fixtures/tiny_qa.csv` (10-row fixture).

The helper `get_data_path()` in `lab.py` reads the `DATA_PATH` environment variable. **Do not modify that helper** — the autograder relies on it.
