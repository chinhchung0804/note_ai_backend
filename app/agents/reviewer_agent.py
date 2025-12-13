"""
Reviewer Agent - Review text bằng CrewAI
Nhiệm vụ: Nhận text từ Text Agent, review và đánh giá chất lượng
"""
import asyncio
import json
from typing import Any, Dict

from crewai import Agent, Crew, Task

from app.agents.llm_config import get_processing_llm

reviewer_agent = Agent(
    role='Text Reviewer',
    goal='Review và đánh giá chất lượng text, tìm lỗi, đưa ra nhận xét',
    backstory=(
        'Bạn là editor chuyên nghiệp với kinh nghiệm nhiều năm review văn bản. '
        'Bạn giỏi phát hiện lỗi chính tả, ngữ pháp, và đánh giá chất lượng nội dung. '
        'Nhiệm vụ của bạn là review text đã được chuẩn hóa và đưa ra đánh giá.'
    ),
    verbose=False,
    llm=get_processing_llm(),
    allow_delegation=False
)


def _build_review_task() -> Task:
    return Task(
        description=(
            "Review đoạn text đã chuẩn hóa và (nếu có) so sánh với bản gốc.\n\n"
            "Text chuẩn hóa:\n{normalized_text}\n\n"
            "Text gốc (có thể rỗng):\n{original_text}\n\n"
            "Nhiệm vụ:\n"
            "1. Nêu lỗi chính tả/ngữ pháp nếu có.\n"
            "2. Đánh giá tính logic, mạch lạc.\n"
            "3. Đề xuất cải thiện cụ thể.\n"
            "4. Chấm điểm chất lượng (tốt/khá/trung bình/kém) và cho biết text có đạt yêu cầu hay không.\n"
            "Trả về JSON với format:\n"
            "{\n"
            '  "valid": true/false,\n'
            '  "quality_score": "tốt/khá/trung bình/kém",\n'
            '  "notes": "nhận xét chi tiết",\n'
            '  "suggestions": ["gợi ý 1", "..."]\n'
            "}"
        ),
        expected_output="JSON đánh giá chất lượng text theo định dạng yêu cầu.",
        agent=reviewer_agent,
        async_execution=False,
    )


def _run_review_task_sync(task: Task, normalized_text: str, original_text: str) -> str:
    crew = Crew(agents=[reviewer_agent], tasks=[task], verbose=False)
    result: Any = crew.kickoff(
        inputs={
            'normalized_text': normalized_text,
            'original_text': original_text,
        }
    )

    if hasattr(result, 'raw') and isinstance(result.raw, str):
        return result.raw
    if isinstance(result, str):
        return result
    return str(result)


async def review_text(normalized_text: str, original_text: str = None) -> Dict[str, Any]:
    """
    Review text và đánh giá chất lượng
    """
    if not normalized_text or normalized_text.strip() == '':
        return {'valid': False, 'notes': 'Text rỗng'}
    
    task = _build_review_task()
    original = original_text or ''

    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(
        None,
        lambda: _run_review_task_sync(task, normalized_text, original)
    )

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # fallback
    return {
        'valid': True,
        'quality_score': 'unknown',
        'notes': raw_output,
        'suggestions': []
    }
