"""DIPPER control-code and seeding helpers.

DIPPER's two knobs exist on opposite scales between the paper and the
pretrained model. The paper reports *diversity* (0 = keep the words,
100 = rewrite everything). The model reads *similarity* (100 = stay
maximally faithful, 0 = fidelity irrelevant). So a desired diversity of X
must be fed to the model as the control code 100 - X.

The DIPPER repository states this directly: "L60-O60 in the paper
corresponds to lex = 40, order = 40 as the control code input to the
model."

Values in configs/dipper_levels.json are therefore expressed as
diversity, and control_codes() is the single place the conversion
happens. This module is deliberately dependency-free so it can be
tested without loading an 11B model.
"""

import zlib

VALID_DIVERSITIES = frozenset({0, 20, 40, 60, 80, 100})

_TORCH_SEED_MODULUS = 2**31


def validate_diversity(value, name):
    """Raise ValueError unless value is a documented DIPPER diversity code."""
    if value not in VALID_DIVERSITIES:
        allowed = ", ".join(str(v) for v in sorted(VALID_DIVERSITIES))
        raise ValueError(
            f"{name} must be one of [{allowed}], got {value!r}. "
            "DIPPER controls are multiples of 20 from 0 to 100."
        )


def control_codes(lex_diversity, order_diversity):
    """Convert desired diversity to the similarity codes the model reads.

    Args:
        lex_diversity: desired lexical diversity, multiple of 20 in [0, 100].
        order_diversity: desired order diversity, multiple of 20 in [0, 100].

    Returns:
        (lex_code, order_code) as similarity values for the DIPPER prompt.
    """
    validate_diversity(lex_diversity, "lexical diversity")
    validate_diversity(order_diversity, "order diversity")
    return int(100 - lex_diversity), int(100 - order_diversity)


def unit_seed(run_seed, source_id, level):
    """Derive a stable seed for one (source_id, level) generation unit.

    Uses zlib.crc32 rather than the builtin hash() because hash() on str is
    salted per process by PYTHONHASHSEED, which would silently break
    reproducibility across runs.

    source_id is normalised to str so a pandas int column and the same id
    read back from CSV derive the same seed.
    """
    key = f"{run_seed}|{source_id}|{level}".encode("utf-8")
    return zlib.crc32(key) % _TORCH_SEED_MODULUS
