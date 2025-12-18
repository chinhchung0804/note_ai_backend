import asyncio
from typing import Any

from crewai import Agent, Crew, Task

from app.agents.llm_config import get_processing_llm

ocr_agent = Agent(
    role='OCR Processing Agent',
    goal='Nhận và cải thiện text từ OCR, sửa lỗi chính tả, loại bỏ ký tự nhiễu',
    backstory=(
        'Bạn là chuyên gia xử lý text OCR với nhiều năm kinh nghiệm. '
        'Bạn giỏi nhận biết và sửa lỗi OCR, đặc biệt với tiếng Việt. '
        'Nhiệm vụ của bạn là nhận raw text từ Tesseract OCR và cải thiện chất lượng.'
    ),
    verbose=False,
    llm=get_processing_llm(),
    allow_delegation=False
)


def _build_ocr_task() -> Task:
    return Task(
        description=(
            "Hãy cải thiện đoạn text sau được sinh ra từ OCR:\n\n"
            "{raw_text}\n\n"
            "Yêu cầu:\n"
            "1. Sửa các lỗi chính tả phổ biến do OCR gây ra.\n"
            "2. Loại bỏ ký tự nhiễu/ký tự lạ.\n"
            "3. Chuẩn hóa khoảng trắng và định dạng câu.\n"
            "4. Giữ nguyên ý nghĩa ban đầu.\n"
            "Chỉ trả về text đã được cải thiện, không giải thích thêm."
        ),
        expected_output="Đoạn văn bản sạch đã được cải thiện từ OCR.",
        agent=ocr_agent,
        async_execution=False,
    )


def _run_task_sync(task: Task, *, raw_text: str) -> str:
    crew = Crew(agents=[ocr_agent], tasks=[task], verbose=False)
    result: Any = crew.kickoff(inputs={'raw_text': raw_text})

    if hasattr(result, 'raw') and isinstance(result.raw, str):
        return result.raw
    if isinstance(result, str):
        return result
    return str(result)


async def process_ocr_text(raw_ocr_text: str) -> str:
    """
    Xử lý text từ OCR, cải thiện chất lượng
    
    Args:
        raw_ocr_text: Text thô từ Tesseract OCR
        
    Returns:
        Text đã được cải thiện chất lượng
    """
    if not raw_ocr_text or raw_ocr_text.strip() == '':
        return ''
    
    task = _build_ocr_task()

    loop = asyncio.get_running_loop()
    cleaned_text = await loop.run_in_executor(
        None,
        lambda: _run_task_sync(task, raw_text=raw_ocr_text)
    )

    return cleaned_text.strip()

