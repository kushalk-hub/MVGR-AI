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
