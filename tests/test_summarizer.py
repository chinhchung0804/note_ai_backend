import json

import pytest

import app.agents.summarizer_agent as sa


@pytest.mark.asyncio
async def test_summarize_text_uses_generated_bundle(monkeypatch):
    async def fake_run_chain(chain, variables):
        if chain is sa.summary_chain:
            return json.dumps(
                {
                    "one_sentence": "Một câu tóm tắt.",
                    "short_paragraph": "Đoạn ngắn 3 câu.",
                    "bullet_points": ["Ý 1", "Ý 2"],
                }
            )
        return json.dumps({})

    monkeypatch.setattr(sa, "_run_chain", fake_run_chain)

    txt = "Hôm nay đội marketing họp về chiến lược mới, ngân sách dự kiến 50 triệu."
    res = await sa.summarize_text(txt, use_rag=False)

    assert isinstance(res, str)
    assert "Đoạn ngắn" in res


@pytest.mark.asyncio
async def test_generate_learning_assets_structure(monkeypatch):
    async def fake_run_chain(chain, variables):
        if chain is sa.summary_chain:
            return json.dumps(
                {
                    "one_sentence": "Một câu.",
                    "short_paragraph": "Ba câu đủ ý.",
                    "bullet_points": ["Điểm 1", "Điểm 2"],
                }
            )
        if chain is sa.question_chain:
            return json.dumps({"questions": [{"question": "Q1", "answer": "A1"}]})
        if chain is sa.mcq_chain:
            return json.dumps(
                {
                    "easy": [
                        {
                            "question": "E1",
                            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                            "answer": "A",
                            "explanation": "exp",
                        }
                    ],
                    "medium": [],
                    "hard": [],
                }
            )
        return "{}"

    monkeypatch.setattr(sa, "_run_chain", fake_run_chain)

    assets = await sa.generate_learning_assets("Dummy text", use_rag=False)

    assert "summaries" in assets and assets["summaries"]["bullet_points"] == ["Điểm 1", "Điểm 2"]
    assert assets["questions"][0]["answer"] == "A1"
    assert "easy" in assets["mcqs"]


@pytest.mark.asyncio
async def test_generate_summary_bundle_fallback(monkeypatch):
    async def failing_run_chain(*args, **kwargs):
        raise RuntimeError("llm error")

    monkeypatch.setattr(sa, "_run_chain", failing_run_chain)

    bundle = await sa.generate_summary_bundle("Chỉ một câu thử nghiệm.", use_rag=False)

    assert "one_sentence" in bundle
    assert bundle["one_sentence"] != ""
