"""Module A tests — stylometric text fingerprinting.

Synthetic text only; no database, no network. spaCy is used when it is
installed, and the "model unavailable" path is exercised explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fingerprinting import pos_features, readability_features, vocabulary_features  # noqa: E402
from fingerprinting.stats_utils import count_syllables  # noqa: E402
from fingerprinting.stylometry import StylometricExtractor, feature_names  # noqa: E402
from shared.preprocessing.text import normalize_text, split_sentences, tokenize  # noqa: E402

LONG_TEXT = " ".join(
    ["The diligent researcher carefully examined every sentence of the document."] * 40
)
PUNCT_HEAVY = "Wait!!! Really??? Yes... no; maybe: (perhaps) -- \"quoted\" / slash."
REPEATED = "cat cat cat cat cat cat cat cat dog dog dog"


def _extract(text: str, **config):
    extractor = StylometricExtractor(config=config)
    return extractor.extract(text)


def test_feature_names_are_stable():
    names = feature_names()
    assert len(names) == 140
    # A second call returns the identical ordered list.
    assert feature_names() == names
    assert len(set(names)) == len(names)


def test_empty_text_is_null_safe():
    features = _extract("")
    # Counts are defined as zero, ratios are undefined (None), never imputed.
    assert features["vocab_total_words"] == 0
    assert features["sent_count"] == 0
    assert features["vocab_type_token_ratio"] is None
    assert features["vocab_avg_word_length"] is None
    assert features["sent_avg_length"] is None
    assert features["read_flesch_reading_ease"] is None


def test_very_short_text():
    features = _extract("Hi.")
    assert features["vocab_total_words"] == 1
    assert features["sent_count"] == 1
    assert features["vocab_type_token_ratio"] == 1.0
    assert features["vocab_hapax_count"] == 1


def test_long_text_is_defined():
    features = _extract(LONG_TEXT)
    assert features["vocab_total_words"] > 300
    assert features["sent_count"] >= 40
    assert 0.0 <= features["vocab_type_token_ratio"] <= 1.0
    assert features["read_flesch_reading_ease"] is not None


def test_punctuation_heavy_text():
    features = _extract(PUNCT_HEAVY)
    assert features["punct_exclamation_count"] >= 3
    assert features["punct_question_count"] >= 3
    assert features["punct_parenthesis_count"] >= 2
    assert features["punct_density"] is not None
    assert 0.0 <= features["punct_density"] <= 1.0


def test_repeated_vocabulary_lowers_diversity():
    repeated = _extract(REPEATED)
    varied = _extract(" ".join(f"token{index}" for index in range(11)))
    assert repeated["vocab_type_token_ratio"] < varied["vocab_type_token_ratio"]
    assert repeated["vocab_unique_words"] == 2


def test_sentence_lengths_differ():
    short = _extract("One. Two. Three. Four. Five.")
    long = _extract("This is a much longer single sentence with many words in it indeed.")
    assert short["sent_count"] == 5
    assert long["sent_count"] == 1
    assert short["sent_length_std"] is not None
    assert long["sent_length_std"] is None  # a single sentence has no spread


def test_missing_spacy_model_degrades_honestly():
    # A model name that cannot exist: POS and syntactic features must be None,
    # the lexical features must still be present.
    extractor = StylometricExtractor(
        config={"spacy_model": "this_model_does_not_exist_zzz", "spacy_disable": ["ner"]}
    )
    assert extractor.spacy_available is False
    assert extractor.status()["spacy_error"] is not None
    features = extractor.extract("A short sentence for the parser.")
    assert features["vocab_total_words"] == 6
    assert features["pos_noun_ratio"] is None
    assert features["syn_tree_depth"] is None


def test_doc_none_matches_shared_tokenizer():
    # Features derived from the shared tokenizer must agree whether or not a
    # spaCy document is supplied.
    text = normalize_text("Alpha beta gamma. Delta epsilon.")
    tokens = tokenize(text)
    sentences = split_sentences(text)
    extractor = StylometricExtractor(config={"spacy_model": "this_model_does_not_exist_zzz"})
    with_doc = extractor.extract(text, tokens=tokens, sentences=sentences, doc=None)
    assert with_doc["vocab_total_words"] == len(tokens)
    assert with_doc["sent_count"] == len(sentences)


def test_vocabulary_module_direct():
    text = normalize_text("one two three four five six seven")
    features = vocabulary_features.extract(
        text, tokens=tokenize(text), sentences=split_sentences(text)
    )
    assert features["vocab_total_words"] == 7
    assert features["vocab_unique_words"] == 7


def test_readability_syllable_heuristic():
    assert count_syllables("cat") == 1
    assert count_syllables("banana") >= 2
    assert count_syllables("") == 0
    text = normalize_text("The cat sat on the mat.")
    features = readability_features.extract(
        text, tokens=tokenize(text), sentences=split_sentences(text)
    )
    assert features["read_avg_syllables_per_word"] is not None
    assert features["read_flesch_kincaid_grade"] is not None


def test_pos_features_without_doc_all_none():
    features = pos_features.extract(
        "", tokens=[], sentences=[], doc=None
    )
    assert features and all(value is None for value in features.values())


@pytest.mark.parametrize("text", ["", "Hi.", PUNCT_HEAVY, LONG_TEXT, REPEATED])
def test_all_features_are_finite_or_none(text):
    import math

    features = _extract(text)
    for name, value in features.items():
        if value is None:
            continue
        assert isinstance(value, (int, float)), name
        assert math.isfinite(float(value)), name
