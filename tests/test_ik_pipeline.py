from __future__ import annotations

from vibeik.cli import run


EXAMPLE_TEXT = (
    "I am using the KUKA KR120 R2500 robot. "
    "I want to move the tooltip of an 8mm drilling tool to [1.5m, 0.1m, 1.0m]"
)


def test_ik_pipeline_returns_response(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = run(EXAMPLE_TEXT)
    assert response.ok in (True, False)
    assert response.meta is not None
