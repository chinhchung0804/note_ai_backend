"""
Shared LLM configuration for CrewAI agents.
Ensures we always instantiate a Gemini-powered LLM with the correct provider.
"""
import os
from typing import Optional

from crewai import LLM

_gemini_llm: Optional[LLM] = None


def _normalize_model_name(raw_model: str) -> str:
    model = (raw_model or '').strip()
    if not model:
        return 'models/gemini-2.0-flash'
    if model.startswith('google/'):
        return model
    if model.startswith('models/'):
        model = model.split('/', 1)[1]
    return f'google/{model}'


def get_gemini_llm() -> LLM:
    """Return a singleton Gemini LLM instance for CrewAI agents."""
    global _gemini_llm
    if _gemini_llm is not None:
        return _gemini_llm

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is required to run CrewAI agents.')

    raw_model = os.getenv('GEMINI_MODEL', 'models/gemini-2.0-flash')
    model = _normalize_model_name(raw_model)

    _gemini_llm = LLM(
        model=model,
        api_key=api_key,
    )
    return _gemini_llm


