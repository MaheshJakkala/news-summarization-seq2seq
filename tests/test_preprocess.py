"""
tests/test_preprocess.py
------------------------
Unit tests for text cleaning and preprocessing utilities.
Run with: pytest tests/ -v
"""

import pytest
from src.preprocess import (
    text_strip,
    wrap_summary_tokens,
    coverage_at_threshold,
    MAX_TEXT_LEN,
    MAX_SUMMARY_LEN,
    START_TOKEN,
    END_TOKEN,
)
import pandas as pd


class TestTextStrip:

    def test_removes_tabs_and_newlines(self):
        result = list(text_strip(["hello\tworld\nfoo\rbar"]))
        assert "\t" not in result[0]
        assert "\n" not in result[0]
        assert "\r" not in result[0]

    def test_removes_repeated_dashes(self):
        result = list(text_strip(["hello --- world"]))[0]
        assert "---" not in result

    def test_removes_repeated_underscores(self):
        result = list(text_strip(["hello ___world"]))[0]
        assert "___" not in result

    def test_removes_special_characters(self):
        result = list(text_strip(["hello <world> (foo) [bar]"]))[0]
        assert "<" not in result
        assert ">" not in result

    def test_lowercases_output(self):
        result = list(text_strip(["Hello World UPPER"]))[0]
        assert result == result.lower()

    def test_normalizes_inc_numbers(self):
        result = list(text_strip(["INC12345 was resolved"]))[0]
        assert "inc_num" in result
        assert "inc12345" not in result

    def test_normalizes_cm_numbers(self):
        result = list(text_strip(["change CM9876 approved"]))[0]
        assert "cm_num" in result

    def test_collapses_multiple_spaces(self):
        result = list(text_strip(["hello     world"]))[0]
        assert "  " not in result

    def test_url_replaced_with_domain(self):
        result = list(text_strip(["visit https://www.google.com/search?q=test now"]))[0]
        assert "www.google.com" in result

    def test_empty_string(self):
        result = list(text_strip([""]))[0]
        assert isinstance(result, str)

    def test_generator_yields_correct_count(self):
        inputs = ["text one", "text two", "text three"]
        outputs = list(text_strip(inputs))
        assert len(outputs) == 3


class TestWrapSummaryTokens:

    def test_adds_start_and_end_tokens(self):
        summaries = ["hello world", "foo bar"]
        result = wrap_summary_tokens(summaries)
        for s in result:
            assert START_TOKEN in s
            assert END_TOKEN in s

    def test_retains_original_content(self):
        summaries = ["breaking news today"]
        result = wrap_summary_tokens(summaries)
        assert "breaking news today" in result[0]

    def test_output_length_matches_input(self):
        summaries = ["a", "b", "c"]
        assert len(wrap_summary_tokens(summaries)) == 3

    def test_start_token_comes_before_end(self):
        result = wrap_summary_tokens(["content"])[0]
        assert result.index(START_TOKEN) < result.index(END_TOKEN)


class TestCoverageAtThreshold:

    def test_full_coverage(self):
        series = pd.Series(["one two", "three", "four five six"])
        ratio = coverage_at_threshold(series, threshold=10)
        assert ratio == 1.0

    def test_zero_coverage(self):
        series = pd.Series(["one two three four five"] * 5)
        ratio = coverage_at_threshold(series, threshold=2)
        assert ratio == 0.0

    def test_partial_coverage(self):
        series = pd.Series(["short", "a bit longer text here"])
        ratio = coverage_at_threshold(series, threshold=3)
        assert 0.0 < ratio < 1.0


class TestConstants:

    def test_max_text_len_value(self):
        assert MAX_TEXT_LEN == 100

    def test_max_summary_len_value(self):
        assert MAX_SUMMARY_LEN == 15

    def test_tokens_are_strings(self):
        assert isinstance(START_TOKEN, str)
        assert isinstance(END_TOKEN, str)

    def test_tokens_are_distinct(self):
        assert START_TOKEN != END_TOKEN
