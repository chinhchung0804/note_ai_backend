def build_output(
    summaries,
    review,
    raw_text=None,
    processed_text=None,
    questions=None,
    mcqs=None,
    sources=None
):
    """
    Chuẩn hóa cấu trúc trả về cho API.
    - `summary`: giữ compatibility (đoạn ngắn 3-5 câu)
    - `summaries`: đầy đủ 3 dạng (1 câu, 3-5 câu, bullet points)
    - `questions`: danh sách câu hỏi tự luận
    - `mcqs`: câu hỏi trắc nghiệm theo độ khó
    """
    if isinstance(summaries, dict):
        primary_summary = summaries.get('short_paragraph') or summaries.get('one_sentence')
    else:
        primary_summary = summaries
    
    result = {
        'summary': primary_summary,
        'summaries': summaries,
        'review': review,
        'questions': questions or [],
        'mcqs': mcqs or {},
        'raw_text': raw_text,
        'processed_text': processed_text or raw_text
    }

    if sources is not None:
        result['sources'] = sources

    return result
