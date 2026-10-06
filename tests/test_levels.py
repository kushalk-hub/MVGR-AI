import pytest

from paraphrase.levels import (
    VALID_DIVERSITIES,
    control_codes,
    unit_seed,
    validate_diversity,
)


def test_paper_headline_pair_maps_to_documented_codes():
    # DIPPER README: "L60-O60 in the paper corresponds to lex = 40, order = 40"
    assert control_codes(60, 60) == (40, 40)


def test_controls_are_the_inverse_of_diversity():
    assert control_codes(0, 0) == (100, 100)
    assert control_codes(20, 0) == (80, 100)
    assert control_codes(40, 20) == (60, 80)
    assert control_codes(60, 40) == (40, 60)
    assert control_codes(100, 100) == (0, 0)


def test_control_codes_validates_its_inputs():
    with pytest.raises(ValueError, match="lexical diversity"):
        control_codes(25, 0)
    with pytest.raises(ValueError, match="order diversity"):
        control_codes(20, 35)


@pytest.mark.parametrize("value", sorted(VALID_DIVERSITIES))
def test_validate_diversity_accepts_the_documented_grid(value):
    validate_diversity(value, "lexical diversity")


@pytest.mark.parametrize("value", [-20, 10, 35, 75, 120])
def test_validate_diversity_rejects_undocumented_values(value):
    with pytest.raises(ValueError):
        validate_diversity(value, "lexical diversity")


def test_unit_seed_is_deterministic():
    # The literal is a regression pin, not a magic number: it locks the
    # zlib.crc32 derivation in place. Swapping crc32 for builtin hash() is
    # invisible to every other test here -- hash() is deterministic within a
    # process, collision-free on these pairs, and in range after the modulo --
    # so without this pin the PYTHONHASHSEED-salted breakage it would cause
    # would reach a GPU run unnoticed.
    assert unit_seed(42, 7, "L2") == 282916338
    assert unit_seed(42, 7, "L2") == unit_seed(42, 7, "L2")


def test_unit_seed_differs_per_level_and_per_source():
    assert unit_seed(42, 7, "L1") != unit_seed(42, 7, "L2")
    assert unit_seed(42, 7, "L2") != unit_seed(42, 8, "L2")


def test_unit_seed_changes_with_run_seed():
    assert unit_seed(42, 7, "L2") != unit_seed(43, 7, "L2")


def test_unit_seed_is_within_torch_seed_range():
    for sid in range(50):
        for level in ("L0", "L1", "L2", "L3", "L4"):
            seed = unit_seed(42, sid, level)
            assert 0 <= seed < 2**31


def test_unit_seed_survives_source_id_type_changes():
    # source_id arrives as int from pandas but as str when read back from CSV
    assert unit_seed(42, 7, "L2") == unit_seed(42, "7", "L2")


def test_ladder_config_is_expressed_as_diversity():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]

    by_level = {entry["level"]: entry for entry in levels}
    assert set(by_level) == {"L1", "L2", "L3", "L4"}

    assert (by_level["L1"]["lex"], by_level["L1"]["order"]) == (20, 0)
    assert (by_level["L2"]["lex"], by_level["L2"]["order"]) == (40, 20)
    assert (by_level["L3"]["lex"], by_level["L3"]["order"]) == (60, 40)
    assert (by_level["L4"]["lex"], by_level["L4"]["order"]) == (60, 60)


def test_ladder_is_monotone_on_both_axes():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]
    ordered = [entry for entry in sorted(levels, key=lambda e: e["level"])]

    lex = [entry["lex"] for entry in ordered]
    order = [entry["order"] for entry in ordered]
    assert lex == sorted(lex)
    assert order == sorted(order)


def test_every_ladder_entry_is_a_valid_diversity_pair():
    import json
    from pathlib import Path

    levels = json.loads(
        Path("configs/dipper_levels.json").read_text(encoding="utf-8")
    )["levels"]
    for entry in levels:
        # raises ValueError on any undocumented value
        control_codes(entry["lex"], entry["order"])


class _FakeTensor(list):
    """paraphrase() calls .to() on each value, as real BatchEncoding values are tensors."""

    def to(self, device):
        return self


class _FakeEncoding(dict):
    """Mimics a BatchEncoding closely enough for paraphrase()."""


class _FakeModel:
    """paraphrase() only touches .device and .generate()."""

    device = "cpu"

    def __init__(self, capture):
        self._capture = capture

    def generate(self, **kwargs):
        return [[0, 1, 2]]


class _FakeTokenizer:
    def __init__(self, capture):
        self._capture = capture

    def __call__(self, texts, **kwargs):
        self._capture["prompt"] = texts[0]
        return _FakeEncoding(input_ids=_FakeTensor([0, 1, 2]))

    def batch_decode(self, ids, **kwargs):
        return ["rewritten"]


def test_paraphrase_prompt_uses_similarity_codes(monkeypatch):
    """The prompt must carry similarity codes, not the raw diversity values."""
    from paraphrase import dipper_generate

    capture = {}
    monkeypatch.setattr(
        dipper_generate, "sent_tokenize", lambda text: ["one two three."]
    )

    out = dipper_generate.paraphrase(
        model=_FakeModel(capture),
        tokenizer=_FakeTokenizer(capture),
        text="one two three.",
        lex_diversity=60,
        order_diversity=60,
        max_length=512,
        top_p=0.75,
        sent_interval=3,
    )

    assert "lexical = 40, order = 40" in capture["prompt"]
    assert "lexical = 60, order = 60" not in capture["prompt"]
    assert out == "rewritten"


def test_paraphrase_rejects_undocumented_diversity(monkeypatch):
    from paraphrase import dipper_generate

    capture = {}
    monkeypatch.setattr(dipper_generate, "sent_tokenize", lambda text: ["a b c."])

    with pytest.raises(ValueError, match="lexical diversity"):
        dipper_generate.paraphrase(
            model=_FakeModel(capture),
            tokenizer=_FakeTokenizer(capture),
            text="a b c.",
            lex_diversity=45,
            order_diversity=0,
            max_length=512,
            top_p=0.75,
            sent_interval=3,
        )
