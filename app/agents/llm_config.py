"""
Shared LLM configuration for CrewAI + LangChain agents with fallback support.

Priority: Gemini (Google) → OpenAI-compatible (MegaLLM, etc.).
"""
import os
from typing import Optional

from crewai import LLM

_gemini_llm: Optional[LLM] = None
_openai_llm: Optional[LLM] = None
_processing_llm: Optional[LLM] = None 
_langchain_llm: Optional[object] = None  
_langchain_fallback_llm: Optional[object] = None 
_langchain_gemini_llm: Optional[object] = None 


def _normalize_model_name(raw_model: str) -> str:
    model = (raw_model or '').strip()
    if not model:
        return 'models/gemini-2.0-flash'
    if model.startswith('google/'):
        return model
    if model.startswith('models/'):
        model = model.split('/', 1)[1]
    return f'google/{model}'


def _build_gemini_llm() -> LLM:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is required to run CrewAI agents.')

    raw_model = os.getenv('GEMINI_MODEL', 'models/gemini-2.0-flash')
    model = _normalize_model_name(raw_model)
    return LLM(model=model, api_key=api_key)


def _build_openai_llm() -> LLM:
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LANGCHAIN_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY or LANGCHAIN_API_KEY is required for OpenAI fallback.')

    model = os.getenv('OPENAI_MODEL', 'openai-gpt-oss-20b')
    base_url = os.getenv('OPENAI_BASE_URL')
    kwargs = {'model': model, 'api_key': api_key}
    if base_url:
        kwargs['base_url'] = base_url
    return LLM(**kwargs)


def get_openai_llm() -> LLM:
    """Return a singleton OpenAI-compatible LLM (MegaLLM, etc.) for CrewAI agents."""
    global _openai_llm
    if _openai_llm is not None:
        return _openai_llm
    _openai_llm = _build_openai_llm()
    return _openai_llm


def get_gemini_llm() -> LLM:
    """
    Return a singleton LLM instance for CrewAI agents.
    Prefers Gemini; falls back to OpenAI-compatible (MegaLLM) if Gemini is unavailable.
    """
    global _gemini_llm
    if _gemini_llm is not None:
        return _gemini_llm

    try:
        _gemini_llm = _build_gemini_llm()
    except Exception as exc:
        print(f"[llm_config] Gemini unavailable, fallback to OpenAI-compatible LLM: {exc}")
        _gemini_llm = get_openai_llm()
    return _gemini_llm


def get_processing_llm() -> LLM:
    """
    CrewAI LLM for preprocessing agents (OCR/Text/Reviewer):
    prefers OpenAI-compatible (MegaLLM) to avoid Gemini quota; falls back to Gemini if missing.
    """
    global _processing_llm
    if _processing_llm is not None:
        return _processing_llm
    try:
        _processing_llm = _build_openai_llm()
    except Exception as exc_primary:
        print(f"[llm_config] Processing LLM: OpenAI primary unavailable, fallback to Gemini: {exc_primary}")
        _processing_llm = _build_gemini_llm()
    return _processing_llm


def _build_gemini_chat_llm(*, temperature: float = 0.2):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is required for Gemini chat model.')
    model = os.getenv('GEMINI_MODEL', 'models/gemini-2.0-flash')
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )


def _build_openai_chat_llm(*, temperature: float = 0.2):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LANGCHAIN_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY or LANGCHAIN_API_KEY is required for OpenAI chat model.')

    model = os.getenv('OPENAI_MODEL', 'openai-gpt-oss-20b')
    base_url = os.getenv('OPENAI_BASE_URL')
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def get_langchain_chat_llm(*, temperature: float = 0.2):
    """
    Shared LangChain chat model (primary = Gemini, fallback = OpenAI-compatible).
    """
    global _langchain_llm
    if _langchain_llm is not None:
        return _langchain_llm
    try:
        _langchain_llm = _build_gemini_chat_llm(temperature=temperature)
    except Exception as exc:
        print(f"[llm_config] Gemini chat unavailable, fallback to OpenAI chat: {exc}")
        _langchain_llm = _build_openai_chat_llm(temperature=temperature)
    return _langchain_llm


def get_openai_chat_llm(*, temperature: float = 0.2):
    """
    Explicit LangChain chat model that uses OpenAI-compatible API (MegaLLM, etc.).
    Useful as a manual fallback without touching the primary singleton.
    """
    global _langchain_fallback_llm
    if _langchain_fallback_llm is not None:
        return _langchain_fallback_llm
    _langchain_fallback_llm = _build_openai_chat_llm(temperature=temperature)
    return _langchain_fallback_llm


def get_gemini_chat_llm(*, temperature: float = 0.2):
    """
    LangChain chat model that forces Gemini (no automatic OpenAI fallback).
    Useful when Gemini must be used for pre/post-processing.
    """
    global _langchain_gemini_llm
    if _langchain_gemini_llm is not None:
        return _langchain_gemini_llm
    _langchain_gemini_llm = _build_gemini_chat_llm(temperature=temperature)
    return _langchain_gemini_llm


