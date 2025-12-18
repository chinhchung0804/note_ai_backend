import asyncio
from typing import Any

from crewai import Agent, Crew, Task

from app.agents.llm_config import get_processing_llm

text_agent = Agent(
    role='Text Processing Agent',
    goal='Chuẩn hóa text, tổ chức lại câu, đoạn văn cho dễ đọc và logic hơn',
    backstory=(
        'Bạn là chuyên gia xử lý và chuẩn hóa văn bản. '
        'Bạn giỏi tách đoạn, sắp xếp lại câu văn cho logic và dễ đọc. '
        'Nhiệm vụ của bạn là nhận text đã được OCR Agent cải thiện và chuẩn hóa thêm.'
    ),
    verbose=False,
    llm=get_processing_llm(),
    allow_delegation=False
)


def _create_normalize_task() -> Task:
    return Task(
        description=(
            "Chuẩn hóa đoạn văn bản sau để đọc dễ hơn. "
            "Tách đoạn hợp lý, sửa dấu câu và đảm bảo mạch lạc.\n\n"
            "Đoạn văn bản:\n{raw_text}\n"
        ),
        expected_output="Trả lại duy nhất đoạn văn bản đã chuẩn hóa.",
        agent=text_agent,
        async_execution=False,
    )


def _run_task_sync(task: Task, raw_text: str) -> str:
    crew = Crew(agents=[text_agent], tasks=[task], verbose=False)
    result: Any = crew.kickoff(inputs={'raw_text': raw_text})

    if hasattr(result, 'raw') and isinstance(result.raw, str):
        return result.raw
    if isinstance(result, str):
        return result
    return str(result)


async def process_and_normalize_text(improved_text: str) -> str:
    """
    Xử lý và chuẩn hóa text
    
    Args:
        improved_text: Text đã được OCR Agent cải thiện
        
    Returns:
        Text đã được chuẩn hóa, tổ chức lại
    """
    if not improved_text or improved_text.strip() == '':
        return ''
    
    task = _create_normalize_task()

    loop = asyncio.get_running_loop()
    normalized = await loop.run_in_executor(
        None,
        lambda: _run_task_sync(task, raw_text=improved_text)
    )

    return normalized.strip()

