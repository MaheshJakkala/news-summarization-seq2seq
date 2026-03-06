"""
preprocess.py
-------------
Text cleaning and dataset preparation for the News Summarization project.

Handles:
- Merging news_summary.csv and news_summary_more.csv
- Regex-based text stripping (URLs, special chars, escape sequences)
- spaCy pipeline for lemmatization/tokenization
- START/END token injection for seq2seq decoder
- Sequence length filtering based on coverage thresholds
"""

import re
import pandas as pd
import numpy as np
from time import time


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_TEXT_LEN = 100     # covers 95.78% of cleaned texts
MAX_SUMMARY_LEN = 15   # covers 99.78% of cleaned summaries

START_TOKEN = "sostok"
END_TOKEN = "eostok"


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_and_merge(summary_path: str, raw_path: str) -> pd.DataFrame:
    """
    Load and merge both CSV files into a unified text/summary DataFrame.

    news_summary_more.csv  → columns: text, headlines
    news_summary.csv       → columns: author, date, read_more, text, ctext, headlines

    For news_summary.csv, the full text is constructed by concatenating:
    author + date + read_more + text + ctext.

    Args:
        summary_path: Path to news_summary.csv
        raw_path:     Path to news_summary_more.csv

    Returns:
        pd.DataFrame with columns ['text', 'summary']
    """
    summary_df = pd.read_csv(summary_path, encoding="iso-8859-1")
    raw_df = pd.read_csv(raw_path, encoding="iso-8859-1")

    pre1 = raw_df.iloc[:, 0:2].copy()

    pre2 = summary_df.iloc[:, 0:6].copy()
    pre2["text"] = (
        pre2["author"]
        .str.cat(pre2["date"], sep=" ")
        .str.cat(pre2["read_more"], sep=" ")
        .str.cat(pre2["text"], sep=" ")
        .str.cat(pre2["ctext"], sep=" ")
    )

    merged = pd.DataFrame()
    merged["text"] = pd.concat([pre1["text"], pre2["text"]], ignore_index=True)
    merged["summary"] = pd.concat([pre1["headlines"], pre2["headlines"]], ignore_index=True)

    return merged


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
def text_strip(column):
    """
    Generator that cleans each row in a column with regex rules:
    - Remove tab/carriage return/newline characters
    - Remove repeated punctuation (__, --, ~~, ++, ..)
    - Remove special characters: < > ( ) | & © ø [ ] ' " , ; ? ~ * !
    - Remove mailto: and hex escape sequences
    - Normalize incident/change ticket numbers (INC###, CM###, CHG###)
    - Remove lone single characters surrounded by spaces
    - Extract domain from URLs (remove path)
    - Collapse multiple spaces

    Args:
        column: iterable of raw text strings

    Yields:
        str: cleaned lowercase text
    """
    for row in column:
        row = re.sub(r"(\t)", " ", str(row)).lower()
        row = re.sub(r"(\r)", " ", str(row)).lower()
        row = re.sub(r"(\n)", " ", str(row)).lower()

        row = re.sub(r"(__+)", " ", str(row)).lower()
        row = re.sub(r"(--+)", " ", str(row)).lower()
        row = re.sub(r"(~~+)", " ", str(row)).lower()
        row = re.sub(r"(\+\++)", " ", str(row)).lower()
        row = re.sub(r"(\.\.+)", " ", str(row)).lower()

        row = re.sub(r"[<>()|&©ø\[\]\'\".,;?~*!]", " ", str(row)).lower()

        row = re.sub(r"(mailto:)", " ", str(row)).lower()
        row = re.sub(r"(\\x9\d)", " ", str(row)).lower()
        row = re.sub(r"([iI][nN][cC]\d+)", "INC_NUM", str(row)).lower()
        row = re.sub(r"([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(row)).lower()

        row = re.sub(r"(\.\s+)", " ", str(row)).lower()
        row = re.sub(r"(\-\s+)", " ", str(row)).lower()
        row = re.sub(r"(\:\s+)", " ", str(row)).lower()
        row = re.sub(r"(\s+.\s+)", " ", str(row)).lower()

        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except Exception:
            pass

        row = re.sub(r"(\s+)", " ", str(row)).lower()
        row = re.sub(r"(\s+.\s+)", " ", str(row)).lower()

        yield row


# ─────────────────────────────────────────────
# spaCy Pipeline
# ─────────────────────────────────────────────
def run_spacy_pipeline(text_gen, batch_size: int = 1000):
    """
    Run spaCy's en_core_web_sm pipeline (NER and parser disabled for speed)
    over a generator of cleaned text strings.

    Args:
        text_gen: generator of cleaned strings
        batch_size: spaCy pipe batch size

    Returns:
        list[str]: tokenized/lemmatized strings
    """
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    t = time()
    result = [str(doc) for doc in nlp.pipe(text_gen, batch_size=batch_size, n_process=-1)]
    print(f"spaCy pipeline done in {round((time() - t) / 60, 2)} mins")
    return result


# ─────────────────────────────────────────────
# Token Wrapping
# ─────────────────────────────────────────────
def wrap_summary_tokens(summary_list: list) -> list:
    """
    Wrap each summary string with START (_START_) and END (_END_) markers,
    then add decoder START/END tokens (sostok / eostok).

    Args:
        summary_list: list of cleaned summary strings

    Returns:
        list[str]: summaries wrapped with sostok ... eostok
    """
    wrapped = [f"_START_ {s} _END_" for s in summary_list]
    wrapped = [f"{START_TOKEN} {s} {END_TOKEN}" for s in wrapped]
    return wrapped


# ─────────────────────────────────────────────
# Sequence Length Filtering
# ─────────────────────────────────────────────
def filter_by_length(
    texts: np.ndarray,
    summaries: np.ndarray,
    max_text_len: int = MAX_TEXT_LEN,
    max_summary_len: int = MAX_SUMMARY_LEN,
) -> pd.DataFrame:
    """
    Keep only pairs where both text ≤ max_text_len and summary ≤ max_summary_len.

    Args:
        texts: cleaned text array
        summaries: cleaned summary array (already wrapped with tokens)
        max_text_len: maximum number of words in text
        max_summary_len: maximum number of words in summary

    Returns:
        pd.DataFrame with columns ['text', 'summary']
    """
    short_text, short_summary = [], []
    for t, s in zip(texts, summaries):
        if len(s.split()) <= max_summary_len and len(t.split()) <= max_text_len:
            short_text.append(t)
            short_summary.append(s)

    print(f"Retained {len(short_text):,} / {len(texts):,} pairs after length filtering")
    return pd.DataFrame({"text": short_text, "summary": short_summary})


# ─────────────────────────────────────────────
# Coverage Analysis Helper
# ─────────────────────────────────────────────
def coverage_at_threshold(series: pd.Series, threshold: int) -> float:
    """
    Return the fraction of samples whose word count is <= threshold.

    Args:
        series: pd.Series of text strings
        threshold: word count threshold

    Returns:
        float: coverage ratio (0–1)
    """
    count = sum(1 for s in series if len(s.split()) <= threshold)
    return count / len(series)
