import os
from typing import Optional

from crewai import LLM

_openai_llm: Optional[LLM] = None
_processing_llm: Optional[LLM] = None
_langchain_llm: Optional[object] = None  
_langchain_fallback_llm: Optional[object] = None


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


def _normalize_model_name(raw_model: str) -> str:
    model = (raw_model or "").strip()
    if not model:
        return "models/gemini-2.0-flash"
    if model.startswith("google/"):
        return model
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    return f"google/{model}"


def _build_gemini_llm() -> LLM:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini LLM.")
    raw_model = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
    model = _normalize_model_name(raw_model)
    return LLM(model=model, api_key=api_key)


def get_openai_llm() -> LLM:
    """Return a singleton OpenAI-compatible LLM (MegaLLM, etc.) for CrewAI agents."""
    global _openai_llm
    if _openai_llm is not None:
        return _openai_llm
    _openai_llm = _build_openai_llm()
    return _openai_llm


def get_processing_llm() -> LLM:
    """
    CrewAI LLM cho OCR/Text/Reviewer: dùng OpenAI.
    """
    global _processing_llm
    if _processing_llm is not None:
        return _processing_llm
    _processing_llm = _build_openai_llm()
    return _processing_llm


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
        timeout=60,  
        max_tokens=4000,  
    )


def get_langchain_chat_llm(*, temperature: float = 0.2):
    """
    Shared LangChain chat model — chỉ dùng OpenAI-compatible (PromptTemplate chains).
    """
    global _langchain_llm
    if _langchain_llm is not None:
        return _langchain_llm
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


def _build_gemini_chat_llm(*, temperature: float = 0.2):
    """Build Gemini LangChain chat model for FREE users"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini chat model.")
    
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        timeout=60,
        max_tokens=4000,
    )


def get_gemini_chat_llm(*, temperature: float = 0.2):
    """
    Get Gemini LangChain chat model for FREE users.
    Uses gemini-2.0-flash-exp (free tier with 20 RPD limit).
    """
    return _build_gemini_chat_llm(temperature=temperature)


def get_chat_llm_for_account(account_type: str, *, temperature: float = 0.2):
    """
    Get appropriate LangChain chat model based on account type.
    
    FREE: GPT-4o-mini (paid, 3 notes/day, basic features)
    PRO: GPT-4 (paid, unlimited, all features)
    ENTERPRISE: GPT-4 (paid, unlimited, all features, priority support)
    """
    from langchain_openai import ChatOpenAI
    from app.database.models import AccountType
    
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LANGCHAIN_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY or LANGCHAIN_API_KEY is required.')
    
    base_url = os.getenv('OPENAI_BASE_URL')
    
    # FREE users: GPT-4o-mini (cheaper, good quality)
    if account_type == AccountType.FREE.value or account_type == AccountType.FREE:
        model = "gpt-4o-mini"
        print(f"[llm_config] Using GPT-4o-mini for FREE account")
    # PRO and ENTERPRISE users: GPT-4 (best quality)
    else:
        model = "gpt-4o-mini"  # Using gpt-4o-mini for now (can change to gpt-4 later)
        print(f"[llm_config] Using GPT-4o-mini for {account_type} account")
    
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=60,
        max_tokens=4000,
    )


