from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from app.core.detector import detect_input_type
from app.core.preprocessor import (
    process_image_file, 
    process_audio_file, 
    process_pdf_file,
    process_docx_file,
    clean_text
)
from app.core.output_builder import build_output
from app.agents.summarizer_agent import (
    generate_learning_assets,
    generate_vocab_bundle,
)
from app.agents.reviewer_agent import review_text
from app.agents.ocr_agent import process_ocr_text
from app.agents.text_agent import process_and_normalize_text
import asyncio

import json
import random


async def process_text(
    text: str,
    db: Optional[Session] = None,
    use_rag: bool = True,
    content_type: Optional[str] = None,
    checked_vocab_items: Optional[str] = None,
):
    """
    Xử lý text input trực tiếp từ người dùng
    Workflow: Clean → Summarize → Build Output
    
    Args:
        text: Text input
        db: Database session (optional, để dùng RAG)
        use_rag: Có sử dụng RAG để cải thiện prompt không (default: True)
    """
    txt = clean_text(text)
    
    if content_type == 'checklist':
        vocab_bundle = await generate_vocab_bundle(txt, checked_vocab_items)
        return build_output(
            summaries=None,
            review={'valid': True, 'notes': 'Vocab checklist từ text'},
            raw_text=txt,
            processed_text=txt,
            questions=[],
            mcqs={},
            vocab_story=vocab_bundle['vocab_story'],
            vocab_mcqs=vocab_bundle['vocab_mcqs'],
            flashcards=vocab_bundle['flashcards'],
            mindmap=vocab_bundle['mindmap'],
            summary_table=vocab_bundle['summary_table'],
            cloze_tests=vocab_bundle.get('cloze_tests'),
            match_pairs=vocab_bundle.get('match_pairs'),
        )

    learning_assets = await generate_learning_assets(
        raw_text=txt,
        db=db,
        file_type='text',
        use_rag=use_rag
    )

    return build_output(
        summaries=learning_assets.get('summaries'),
        review={'valid': True, 'notes': 'Text input trực tiếp'},
        raw_text=txt,
        processed_text=txt,
        questions=learning_assets.get('questions'),
        mcqs=learning_assets.get('mcqs')
    )

async def _extract_text_from_upload(upload_file):
    """
    Helper để đọc và chuẩn hóa text từ UploadFile.
    Trả về dict chứa raw_text, processed_text, review_result, input_type, error.
    """
    import tempfile
    import os
    
    suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ''
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    contents = await upload_file.read()
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    file_path = tmp.name
    
    try:
        input_type = detect_input_type(file_path)
        
        raw_text = ''
        error_message = ''
        if input_type == 'image':
            raw_text, error_message = process_image_file(file_path)
        elif input_type == 'audio':
            raw_text = process_audio_file(file_path)
        elif input_type == 'pdf':
            raw_text = process_pdf_file(file_path)
        elif input_type == 'docx':
            raw_text = process_docx_file(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
            except:
                raw_text = ''
        
        raw_text = clean_text(raw_text)
        
        if not raw_text or raw_text.strip() == '':
            if error_message:
                error_note = f"Không thể extract text từ file. Chi tiết: {error_message}"
            else:
                error_note = "File không chứa text hoặc không thể đọc được"
            
            return {
                'input_type': input_type,
                'raw_text': '',
                'processed_text': '',
                'review': {'valid': False, 'notes': error_note, 'error': error_message},
                'error': error_note
            }

        processed_text = raw_text
        review_result = None
        
        if input_type in ('image', 'audio'):
            improved_text = await process_ocr_text(raw_text)
            normalized_text = await process_and_normalize_text(improved_text)
            review_result = await review_text(normalized_text, raw_text)
            processed_text = normalized_text
        else:
            normalized_text = await process_and_normalize_text(raw_text)
            review_result = await review_text(normalized_text, raw_text)
            processed_text = normalized_text
        
        return {
            'input_type': input_type,
            'raw_text': raw_text,
            'processed_text': processed_text,
            'review': review_result,
            'error': None
        }
    finally:
        try:
            os.remove(file_path)
        except:
            pass
        await upload_file.seek(0)


async def process_file(
    upload_file,
    db: Optional[Session] = None,
    use_rag: bool = True,
    content_type: Optional[str] = None,
    checked_vocab_items: Optional[str] = None,
):
    """
    Xử lý file upload đơn lẻ (giữ behaviour cũ để không phá API hiện tại)
    """
    extracted = await _extract_text_from_upload(upload_file)

    if not extracted['processed_text']:
        return build_output(
            summaries='Không thể extract text từ file',
            review=extracted['review'],
            raw_text='',
            processed_text=''
        )

    if content_type == 'checklist':
        vocab_bundle = await generate_vocab_bundle(
            extracted['processed_text'],
            checked_vocab_items
        )
        return build_output(
            summaries=None,
            review=extracted['review'],
            raw_text=extracted['raw_text'],
            processed_text=extracted['processed_text'],
            questions=[],
            mcqs={},
            vocab_story=vocab_bundle['vocab_story'],
            vocab_mcqs=vocab_bundle['vocab_mcqs'],
            flashcards=vocab_bundle['flashcards'],
            mindmap=vocab_bundle['mindmap'],
            summary_table=vocab_bundle['summary_table'],
            cloze_tests=vocab_bundle.get('cloze_tests'),
            match_pairs=vocab_bundle.get('match_pairs'),
        )

    learning_assets = await generate_learning_assets(
        raw_text=extracted['processed_text'],
        db=db,
        file_type=extracted['input_type'],
        use_rag=use_rag
    )

    return build_output(
        summaries=learning_assets.get('summaries'),
        review=extracted['review'],
        raw_text=extracted['raw_text'],
        processed_text=extracted['processed_text'],
        questions=learning_assets.get('questions'),
        mcqs=learning_assets.get('mcqs')
    )


async def process_combined_inputs(
    text_note: Optional[str],
    files: Optional[List],
    db: Optional[Session] = None,
    use_rag: bool = True,
    content_type: Optional[str] = None,
    checked_vocab_items: Optional[str] = None,
):
    """
    Xử lý đồng thời nhiều nguồn input (text + nhiều file) rồi gộp lại thành một ghi chú hoàn chỉnh.
    """
    files = files or []
    sources: List[Dict] = []
    combined_chunks: List[str] = []

    if text_note and text_note.strip():
        cleaned_text = clean_text(text_note)
        sources.append({
            'type': 'text',
            'source': 'note_body',
            'raw_text': text_note,
            'processed_text': cleaned_text,
            'review': {'valid': True, 'notes': 'Input từ ghi chú'}
        })
        combined_chunks.append(cleaned_text)

    for upload_file in files:
        extracted = await _extract_text_from_upload(upload_file)
        source_entry = {
            'type': extracted['input_type'],
            'source': upload_file.filename,
            'raw_text': extracted['raw_text'],
            'processed_text': extracted['processed_text'],
            'review': extracted['review']
        }
        if extracted['error']:
            source_entry['error'] = extracted['error']
        sources.append(source_entry)

        if extracted['processed_text']:
            label = upload_file.filename or extracted['input_type']
            combined_chunks.append(f"[Source: {label}]\\n{extracted['processed_text']}")

    if not combined_chunks:
        return build_output(
            summaries='Không tìm thấy nội dung hợp lệ để xử lý',
            review={'valid': False, 'notes': 'Tất cả inputs đều trống hoặc không đọc được'},
            raw_text='',
            processed_text='',
            sources=sources
        )

    combined_text = "\\n\\n".join(combined_chunks)

    learning_assets = await generate_learning_assets(
        raw_text=combined_text,
        db=db,
        file_type='combined',
        use_rag=use_rag
    )

    if content_type == 'checklist':
        vocab_bundle = await generate_vocab_bundle(
            combined_text,
            checked_vocab_items
        )
        return build_output(
            summaries=learning_assets.get('summaries'),
            review={'valid': True, 'notes': 'Kết hợp nhiều nguồn input - checklist'},
            raw_text=combined_text,
            processed_text=combined_text,
            questions=learning_assets.get('questions'),
            mcqs=learning_assets.get('mcqs'),
            sources=sources,
            vocab_story=vocab_bundle['vocab_story'],
            vocab_mcqs=vocab_bundle['vocab_mcqs'],
            flashcards=vocab_bundle['flashcards'],
            mindmap=vocab_bundle['mindmap'],
            summary_table=vocab_bundle['summary_table'],
            cloze_tests=vocab_bundle.get('cloze_tests'),
            match_pairs=vocab_bundle.get('match_pairs'),
        )

    return build_output(
        summaries=learning_assets.get('summaries'),
        review={'valid': True, 'notes': 'Kết hợp nhiều nguồn input'},
        raw_text=combined_text,
        processed_text=combined_text,
        questions=learning_assets.get('questions'),
        mcqs=learning_assets.get('mcqs'),
        sources=sources
    )


