import json

import pytest

from paraphrase import dipper_generate


def test_build_manifest_records_the_run_configuration():
    manifest = dipper_generate.build_manifest(
        run_seed=42,
        model="test/model",
        tokenizer="test/tokenizer",
        levels=[
            {"level": "L1", "lex": 20, "order": 0},
            {"level": "L2", "lex": 40, "order": 20},
            {"level": "L3", "lex": 60, "order": 40},
            {"level": "L4", "lex": 60, "order": 60},
        ],
        top_p=0.75,
        max_length=512,
        sent_interval=3,
        tally={"ok": 1000, "empty": 0, "error": 0},
    )

    assert manifest["seed"] == 42
    assert manifest["model"] == "test/model"
    assert manifest["tokenizer"] == "test/tokenizer"
    assert manifest["load_in_8bit"] is True
    assert manifest["top_p"] == 0.75
    assert manifest["max_length"] == 512
    assert manifest["sent_interval"] == 3
    assert manifest["tally"] == {"ok": 1000, "empty": 0, "error": 0}

    # Levels are recorded as diversity AND as the codes the model read.
    assert manifest["levels"][0]["lex_diversity"] == 20
    assert manifest["levels"][0]["lex_code"] == 80

    for field in ("torch", "transformers", "bitsandbytes", "generated_at"):
        assert field in manifest


def test_tally_counts_a_mixed_run():
    counts = dipper_generate.Tally()
    counts.record("ok")
    counts.record("ok")
    counts.record("empty")
    counts.record("error")
    assert counts.as_dict() == {"ok": 2, "empty": 1, "error": 1}


def test_tally_starts_at_zero():
    assert dipper_generate.Tally().as_dict() == {"ok": 0, "empty": 0, "error": 0}