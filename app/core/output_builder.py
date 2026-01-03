def build_output(
    summaries,
    review,
    raw_text=None,
    processed_text=None,
    questions=None,
    mcqs=None,
    sources=None,
    vocab_story=None,
    vocab_mcqs=None,
    flashcards=None,
    mindmap=None,
    summary_table=None,
    cloze_tests=None,
    match_pairs=None,
    text_summary=None,
    files_summaries=None,
):
    """
    Chuẩn hóa cấu trúc trả về cho API.
    - `summary`: giữ compatibility (đoạn ngắn 3-5 câu)
    - `summaries`: đầy đủ 3 dạng (1 câu, 3-5 câu, bullet points)
    - `text_summary`: summary riêng cho text note (khi có cả text + file)
    - `files_summaries`: array summary riêng cho từng file (khi có cả text + file)
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
    if vocab_story is not None:
        result['vocab_story'] = vocab_story
    if vocab_mcqs is not None:
        result['vocab_mcqs'] = vocab_mcqs
    if flashcards is not None:
        result['flashcards'] = flashcards
    if mindmap is not None:
        result['mindmap'] = mindmap
    if summary_table is not None:
        result['summary_table'] = summary_table
    if cloze_tests is not None:
        result['cloze_tests'] = cloze_tests
    if match_pairs is not None:
        result['match_pairs'] = match_pairs
    if text_summary is not None:
        result['text_summary'] = text_summary
    if files_summaries is not None:
        result['files_summaries'] = files_summaries

    return result
