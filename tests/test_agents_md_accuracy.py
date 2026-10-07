from pathlib import Path

AGENTS = Path("AGENTS.md")


def read():
    return AGENTS.read_text(encoding="utf-8")


def test_agents_md_exists():
    assert AGENTS.exists()


def test_stale_purely_lexical_claim_is_gone():
    body = read()
    assert "L1-L3 are lex 20/40/60 with order 0" not in body
    assert "purely lexical" not in body


def test_agents_md_records_the_scale_trap():
    body = read().lower()
    assert "diversity" in body
    assert "similarity" in body


def test_agents_md_documents_the_ladder():
    body = read()
    assert "| L1 | 20 | 0 | 80, 100 |" in body
    assert "| L4 | 60 | 60 | 40, 40 |" in body


def test_agents_md_documents_setup_and_seeding():
    body = read()
    assert "setup_env.sh" in body
    assert "preflight" in body.lower()
    assert "seed" in body.lower()


def test_agents_md_keeps_the_pipeline_order_section():
    assert "## Pipeline order" in read()
