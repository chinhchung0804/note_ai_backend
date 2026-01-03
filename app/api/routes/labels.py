"""
Label management routes
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import User
from app.auth.security import get_current_active_user
from app.agents.label_suggester import suggest_labels_for_note, get_popular_labels_by_category


router = APIRouter(prefix="/labels", tags=["Labels"])


# Pydantic schemas
class LabelSuggestionRequest(BaseModel):
    text: str
    existing_labels: Optional[List[str]] = None


class LabelSuggestion(BaseModel):
    category: str
    label: str
    confidence: float
    reason: str
    color: str
    icon: str


class LabelSuggestionResponse(BaseModel):
    suggested_labels: List[LabelSuggestion]
    recommended_categories: List[str]
    is_pro_feature: bool = True
    upgrade_required: bool = False
    upgrade_message: Optional[str] = None


@router.post("/suggest", response_model=LabelSuggestionResponse)
async def suggest_labels(
    request: LabelSuggestionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Suggest labels for note content using AI (PRO only).
    
    **PRO Feature**: This endpoint is only available for PRO and ENTERPRISE users.
    
    **Request:**
    - text: Note content (will be truncated to 1000 chars)
    - existing_labels: User's existing labels (optional, for consistency)
    
    **Response:**
    - suggested_labels: List of AI-suggested labels with confidence scores
    - recommended_categories: Suggested categories
    - upgrade_required: True if user needs to upgrade
    """
    result = await suggest_labels_for_note(
        text=request.text,
        existing_labels=request.existing_labels,
        account_type=current_user.account_type.value
    )
    
    # If upgrade required, return 403 with upgrade message
    if result.get("upgrade_required"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": result.get("error"),
                "upgrade_message": result.get("upgrade_message"),
                "upgrade_required": True
            }
        )
    
    return result


@router.get("/popular")
async def get_popular_labels():
    """
    Get popular labels grouped by category.
    
    Useful for:
    - Autocomplete suggestions
    - Label picker UI
    - Quick label selection
    
    **Available for all users (FREE and PRO)**
    """
    return {
        "categories": get_popular_labels_by_category(),
        "total_categories": len(get_popular_labels_by_category())
    }


@router.get("/categories")
async def get_label_categories():
    """
    Get all available label categories.
    
    **Available for all users (FREE and PRO)**
    """
    categories = list(get_popular_labels_by_category().keys())
    return {
        "categories": categories,
        "total": len(categories)
    }
