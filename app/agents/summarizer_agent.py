import os
import json
import re
import asyncio
from typing import Optional, Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import Session

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = os.getenv('GEMINI_MODEL', 'models/gemini-2.0-flash')

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

summary_prompt_template = PromptTemplate(
    input_variables=['instructions', 'raw_text'],
    template=(
        "{instructions}\n\n"
        "NHIỆM VỤ: Tạo JSON theo đúng schema sau (CHỈ trả về JSON, không có markdown, không có giải thích):\n"
        "{{\n"
        '  "one_sentence": "Tóm tắt 1 câu",\n'
        '  "short_paragraph": "Tóm tắt 3-5 câu",\n'
        '  "bullet_points": ["Ý 1", "Ý 2", "Ý 3"]\n'
        "}}\n"
        "QUAN TRỌNG: Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm.\n\n"
        "NỘI DUNG GỐC:\n{raw_text}\n"
    )
)

question_prompt_template = PromptTemplate(
    input_variables=['raw_text'],
    template=(
        "Bạn là giáo viên chuyên tạo câu hỏi ôn tập chất lượng cao. Dựa vào ghi chú sau, tạo 5-10 câu hỏi tự luận"
        " giúp người học hiểu sâu và nhớ nội dung.\n\n"
        "Yêu cầu:\n"
        "- Mỗi câu hỏi phải kiểm tra hiểu biết, không chỉ nhớ máy móc\n"
        "- Câu hỏi đa dạng: có câu hỏi về khái niệm, so sánh, phân tích, áp dụng\n"
        "- Câu hỏi phải rõ ràng, cụ thể, không mơ hồ\n"
        "- Đáp án phải ngắn gọn nhưng đầy đủ thông tin quan trọng (2-4 câu)\n"
        "- Không được tạo câu hỏi chỉ lặp lại nguyên văn nội dung ghi chú\n\n"
        "Trả về JSON đúng schema (CHỈ JSON, không có markdown):\n"
        "{{\n"
        '  "questions": [\n'
        '    {{"question": "Câu hỏi kiểm tra hiểu biết về khái niệm/ý chính", "answer": "Đáp án ngắn gọn nhưng đầy đủ (2-4 câu)"}},\n'
        '    {{"question": "Câu hỏi yêu cầu so sánh/phân tích", "answer": "Đáp án chi tiết"}},\n'
        '    {{"question": "Câu hỏi về ứng dụng/thực tế", "answer": "Đáp án cụ thể"}}\n'
        "  ]\n"
        "}}\n"
        "QUAN TRỌNG: Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm.\n\n"
        "Ghi chú:\n{raw_text}\n"
    )
)

mcq_prompt_template = PromptTemplate(
    input_variables=['raw_text'],
    template=(
        "Bạn là giáo viên chuyên tạo câu hỏi trắc nghiệm chất lượng cao. Dựa vào ghi chú sau, tạo câu hỏi trắc nghiệm với các yêu cầu:\n"
        "- Mỗi độ khó (easy, medium, hard) tạo từ 1-3 câu hỏi\n"
        "- Mỗi câu hỏi phải kiểm tra hiểu biết về nội dung ghi chú, không chỉ nhớ máy móc\n"
        "- Mỗi câu có 4 phương án A, B, C, D\n"
        "- Đáp án đúng phải phân bố đều (không phải tất cả đều A)\n"
        "- Các phương án sai (distractors) phải:\n"
        "  + Có vẻ hợp lý và liên quan đến chủ đề\n"
        " + Dựa trên thông tin trong ghi chú nhưng sai hoặc không chính xác\n"
        "  + Không được quá rõ ràng là sai\n"
        "- Explanation phải giải thích tại sao đáp án đúng và tại sao các phương án khác sai\n\n"
        "Trả về JSON đúng schema (CHỈ JSON, không có markdown):\n"
        "{{\n"
        '  "easy": [\n'
        '    {{"question": "Câu hỏi kiểm tra kiến thức cơ bản", "options": {{"A": "Đáp án đúng (chi tiết)", "B": "Phương án sai nhưng hợp lý", "C": "Phương án sai nhưng liên quan", "D": "Phương án sai nhưng có vẻ đúng"}}, "answer": "A", "explanation": "Giải thích tại sao A đúng và B, C, D sai"}}\n'
        "  ],\n"
        '  "medium": [\n'
        '    {{"question": "Câu hỏi yêu cầu hiểu sâu hơn", "options": {{"A": "Phương án sai", "B": "Đáp án đúng", "C": "Phương án sai", "D": "Phương án sai"}}, "answer": "B", "explanation": "Giải thích chi tiết"}}\n'
        "  ],\n"
        '  "hard": [\n'
        '    {{"question": "Câu hỏi phân tích, so sánh hoặc áp dụng", "options": {{"A": "Phương án sai", "B": "Phương án sai", "C": "Đáp án đúng", "D": "Phương án sai"}}, "answer": "C", "explanation": "Giải thích chi tiết"}}\n'
        "  ]\n"
        "}}\n"
        "QUAN TRỌNG:\n"
        "- Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm\n"
        "- Đảm bảo đáp án đúng phân bố đều giữa A, B, C, D\n"
        "- Các phương án sai phải có nội dung thực tế, không được chung chung như 'Nội dung hoàn toàn khác'\n\n"
        "Ghi chú:\n{raw_text}\n"
    )
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template)
question_chain = LLMChain(llm=llm, prompt=question_prompt_template)
mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt_template)

# Patterns để extract JSON từ response
JSON_BLOCK_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.S | re.M)
MARKDOWN_JSON_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S | re.M)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def _extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON block từ text response của LLM.
    Hỗ trợ cả markdown code blocks và raw JSON.
    """
    if not text:
        return None
    
    # Thử extract từ markdown code block trước
    markdown_match = MARKDOWN_JSON_PATTERN.search(text)
    if markdown_match:
        return markdown_match.group(1)
    
    # Thử extract JSON block thông thường
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        return match.group()
    
    return None


def _safe_json_loads(payload: str, fallback: Any) -> Any:
    """
    Parse JSON từ LLM response một cách an toàn.
    Hỗ trợ cả markdown code blocks và raw JSON.
    """
    if not payload:
        return fallback
    
    # Thử parse trực tiếp
    try:
        return json.loads(payload.strip())
    except json.JSONDecodeError:
        pass
    
    # Thử extract JSON block (có thể trong markdown hoặc có text thêm)
    json_block = _extract_json_block(payload)
    if json_block:
        try:
                return json.loads(json_block)
        except json.JSONDecodeError as e:
            # Log để debug nhưng không raise
            print(f"[summarizer] JSON parse error: {e}")
            print(f"[summarizer] Extracted block (first 200 chars): {json_block[:200]}")
    
    return fallback


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = SENTENCE_SPLIT_PATTERN.split(text.replace('\n', ' '))
    return [s.strip() for s in sentences if s.strip()]


def _fallback_summary(raw_text: str) -> Dict[str, Any]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        return {
            'one_sentence': raw_text[:200],
            'short_paragraph': raw_text[:300],
            'bullet_points': [raw_text[:200]]
        }
    
    one_sentence = sentences[0]
    short_paragraph = ' '.join(sentences[:3])
    bullet_points = sentences[:5]
    
    return {
        'one_sentence': one_sentence,
        'short_paragraph': short_paragraph,
        'bullet_points': bullet_points
    }


def _fallback_questions(raw_text: str) -> List[Dict[str, str]]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text[:200]]
    
    questions = []
    for idx, sentence in enumerate(sentences[:10], 1):
        questions.append({
            'question': f"Nội dung quan trọng số {idx} là gì liên quan tới: \"{sentence}\"?",
            'answer': sentence
        })
    
    while len(questions) < 5:
        questions.append({
            'question': f"Hãy nêu lại ý chính số {len(questions)+1} của ghi chú.",
            'answer': sentences[0]
        })
    
    return questions


def _fallback_mcqs(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text[:200]]
    
    levels = ['easy', 'medium', 'hard']
    mcqs: Dict[str, List[Dict[str, Any]]] = {}
    option_labels = ['A', 'B', 'C', 'D']
    for idx, level in enumerate(levels):
        fact = sentences[min(idx, len(sentences) - 1)]
        distractors = []
        for offset in range(1, 4):
            source_idx = min(idx + offset, len(sentences) - 1)
            distractor_sentence = sentences[source_idx]
            if distractor_sentence == fact:
                distractor_sentence = f"Cách giải thích khác cho nội dung: {fact}"
            distractors.append(distractor_sentence)
        options = {}
        correct_label = option_labels[idx % len(option_labels)]
        distractor_iter = iter(distractors)
        for label in option_labels:
            if label == correct_label:
                options[label] = fact
            else:
                options[label] = next(distractor_iter)
        mcqs[level] = [{
            'question': f"Dựa trên ghi chú, ý nào mô tả chính xác nhất: \"{fact}\"?",
            'options': options,
            'answer': correct_label,
            'explanation': (
                f"Phương án {correct_label} tái hiện đúng nội dung đã nêu. "
                "Các phương án khác đề cập chi tiết liên quan nhưng không khớp hoàn toàn với ý chính này."
            )
        }]
    return mcqs


async def _run_chain(chain: LLMChain, variables: Dict[str, Any]) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: chain.run(variables))


def _build_summary_instructions(
    db: Optional[Session],
    raw_text: str,
    file_type: Optional[str],
    use_rag: bool
) -> str:
    instructions = (
        "Bạn là trợ lý AI giúp tạo tài liệu ôn tập tiếng Việt."
        " Hãy sửa lỗi chính tả nếu cần và nêu rõ thông tin trọng tâm."
    )
    
    if file_type:
        instructions += f"\nNgữ cảnh: nội dung được trích từ {file_type.upper()}."
    
    if db and use_rag:
        try:
            from app.services.prompt_retriever import prompt_retriever
            rag_prompt = prompt_retriever.get_contextual_prompt(
                db=db,
                raw_text=raw_text,
                file_type=file_type
            )
            instructions = rag_prompt.strip()
        except Exception as exc:
            print(f"[summarizer] Skip RAG prompt due to error: {exc}")
    
    instructions += (
        "\nĐảm bảo:\n"
        "- one_sentence <= 40 từ\n"
        "- short_paragraph dài 3-5 câu, giữ số liệu quan trọng\n"
        "- bullet_points từ 3-7 ý, mỗi ý ngắn gọn"
    )
    return instructions


async def generate_summary_bundle(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    instructions = _build_summary_instructions(db, raw_text, file_type, use_rag)
    try:
        response = await _run_chain(
            summary_chain,
            {'instructions': instructions, 'raw_text': raw_text}
        )
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict):
            bullets = parsed.get('bullet_points')
            if isinstance(bullets, str):
                parsed['bullet_points'] = [b.strip() for b in bullets.split('\n') if b.strip()]
            return parsed
    except Exception as exc:
        print(f"[summarizer] Error generating summaries: {exc}")
    
    return _fallback_summary(raw_text)


async def generate_question_set(raw_text: str) -> List[Dict[str, str]]:
    try:
        response = await _run_chain(question_chain, {'raw_text': raw_text})
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict) and isinstance(parsed.get('questions'), list):
            return parsed['questions']
    except Exception as exc:
        print(f"[summarizer] Error generating questions: {exc}")
    return _fallback_questions(raw_text)


async def generate_mcq_set(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        response = await _run_chain(mcq_chain, {'raw_text': raw_text})
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict):
            normalized = {}
            for level in ['easy', 'medium', 'hard']:
                level_questions = parsed.get(level, [])
                if not isinstance(level_questions, list):
                    continue
                normalized[level] = level_questions
            if normalized:
                return normalized
    except Exception as exc:
        print(f"[summarizer] Error generating MCQs: {exc}")
    return _fallback_mcqs(raw_text)


async def generate_learning_assets(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    summaries = await generate_summary_bundle(
        raw_text=raw_text,
        db=db,
        file_type=file_type,
        use_rag=use_rag
    )
    questions = await generate_question_set(raw_text)
    mcqs = await generate_mcq_set(raw_text)
    return {
        'summaries': summaries,
        'questions': questions,
        'mcqs': mcqs
    }


async def summarize_text(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> str:
    """
    Backwards-compatible helper để lấy đoạn tóm tắt chính (3-5 câu).
    """
    summaries = await generate_summary_bundle(
        raw_text=raw_text,
        db=db,
        file_type=file_type,
        use_rag=use_rag
    )
    return summaries.get('short_paragraph') or summaries.get('one_sentence') or ''
