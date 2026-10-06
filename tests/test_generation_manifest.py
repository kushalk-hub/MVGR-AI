import json
import sys

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


def test_tally_rejects_an_unknown_status():
    with pytest.raises(KeyError):
        dipper_generate.Tally().record("skipped")


def _row(source_id, level, label, status):
    return {
        "sample_id": f"{source_id}_{level}",
        "source_id": source_id,
        "text": "some text",
        "label": label,
        "paraphrase_level": level,
        "lex_control": "",
        "order_control": "",
        "parent_sample_id": "",
        "quality_status": status,
    }


def test_dataset_tally_excludes_human_and_seeded_l0_rows(tmp_path):
    path = tmp_path / "pilot_dataset.csv"
    dipper_generate.write_rows(str(path), [
        _row("h1", "human", "human", "ok"),
        _row("h2", "human", "human", "ok"),
        _row("s1", "L0", "ai", "ok"),
        _row("s2", "L0", "ai", "ok"),
        _row("s1", "L1", "ai", "ok"),
        _row("s2", "L1", "ai", "error"),
        _row("s3", "L2", "ai", "empty"),
    ])

    assert dipper_generate.tally_dataset(str(path)) == {"ok": 1, "empty": 1, "error": 1}


class _StubTokenizer:
    @staticmethod
    def from_pretrained(name):
        return object()


class _StubModel:
    device = "cpu"

    def eval(self):
        return self


def _stub_generation(monkeypatch, tmp_path, outcomes):
    """Drive main() with no DIPPER load; `outcomes` is consumed one per unit."""
    source = tmp_path / "pilot_source.csv"
    source.write_text(
        "source_id,text,label\n"
        "s1,alpha beta gamma.,ai\n"
        "s2,delta epsilon zeta.,ai\n"
        "h1,a human sentence.,human\n",
        encoding="utf-8",
    )

    steps = iter(outcomes)

    def fake_paraphrase(model, tokenizer, text, lex, order, max_length, top_p, sent_interval):
        outcome = next(steps)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(dipper_generate, "ensure_punkt", lambda: None)
    monkeypatch.setattr(dipper_generate, "T5Tokenizer", _StubTokenizer)
    monkeypatch.setattr(dipper_generate, "load_model", lambda model_id: _StubModel())
    monkeypatch.setattr(dipper_generate, "paraphrase", fake_paraphrase)
    monkeypatch.setattr(
        dipper_generate,
        "load_levels",
        lambda path: [
            {"level": "L1", "lex": 20, "order": 0},
            {"level": "L2", "lex": 40, "order": 20},
        ],
    )
    monkeypatch.setattr(sys, "argv", [
        "dipper_generate.py",
        "--input", str(source),
        "--output", str(tmp_path / "pilot_dataset.csv"),
        "--seed", "42",
    ])


def test_manifest_tally_survives_a_resumed_run(tmp_path, monkeypatch, capsys):
    """A second run has nothing to generate; the manifest must not read clean."""
    _stub_generation(
        monkeypatch,
        tmp_path,
        ["rewritten a", "", RuntimeError("boom"), "rewritten b"],
    )
    dipper_generate.main()

    manifest = json.loads(
        (tmp_path / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tally"] == {"ok": 2, "empty": 1, "error": 1}

    _stub_generation(monkeypatch, tmp_path, [])
    dipper_generate.main()

    rows_after = capsys.readouterr().out
    assert "ok=0 empty=0 error=0" in rows_after

    resumed = json.loads(
        (tmp_path / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert resumed["tally"] == {"ok": 2, "empty": 1, "error": 1}
