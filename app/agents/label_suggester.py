"""
AI Label Suggestion Agent (PRO Feature)
Automatically suggests labels/categories for notes based on content
"""
import json
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.agents.llm_config import get_chat_llm_for_account


LABEL_SUGGESTION_PROMPT = PromptTemplate(
    input_variables=['text', 'existing_labels'],
    template=(
        "Bạn là AI chuyên phân loại và gợi ý nhãn (labels) cho ghi chú học tập.\n\n"
        
        "NHIỆM VỤ: Phân tích nội dung ghi chú và gợi ý các labels phù hợp.\n\n"
        
        "QUY TẮC:\n"
        "1. Gợi ý 3-5 labels quan trọng nhất\n"
        "2. Mỗi label có category (Môn học, Địa điểm, Chủ đề, Cấp độ, Kỹ năng)\n"
        "3. Confidence score từ 0-1 (chỉ gợi ý nếu confidence > 0.7)\n"
        "4. Labels phải ngắn gọn (1-3 từ), dễ hiểu\n"
        "5. Ưu tiên labels phổ biến, dễ tìm kiếm\n"
        "6. Nếu có existing labels, ưu tiên sử dụng lại (để consistency)\n\n"
        
        "CATEGORIES PHỔ BIẾN:\n"
        "- Môn học: Toán, Lý, Hóa, Văn, Anh, Sử, Địa, Sinh, GDCD, Tin học\n"
        "- Địa điểm: Tên thành phố, quốc gia, địa danh (Huế, Đà Nẵng, Hà Nội, etc.)\n"
        "- Chủ đề: Du lịch, Ẩm thực, Công nghệ, Kinh doanh, Sức khỏe, Thể thao, Nghệ thuật\n"
        "- Cấp độ: Tiểu học, THCS, THPT, Đại học, Cao học\n"
        "- Kỹ năng: Nghe, Nói, Đọc, Viết, Ngữ pháp, Từ vựng, Phát âm\n"
        "- Ngôn ngữ: Tiếng Anh, Tiếng Nhật, Tiếng Hàn, Tiếng Trung\n\n"
        
        "NỘI DUNG GHI CHÚ:\n"
        "{text}\n\n"
        
        "EXISTING LABELS CỦA USER (tham khảo để consistency):\n"
        "{existing_labels}\n\n"
        
        "Trả về JSON (CHỈ JSON, không có markdown):\n"
        "{{\n"
        '  "suggested_labels": [\n'
        '    {{\n'
        '      "category": "Category name",\n'
        '      "label": "Label name",\n'
        '      "confidence": 0.95,\n'
        '      "reason": "Lý do ngắn gọn (1 câu)"\n'
        '    }}\n'
        "  ],\n"
        '  "recommended_categories": ["Category 1", "Category 2"]\n'
        "}}\n"
    )
)


def _safe_json_loads(text: str, fallback: Any) -> Any:
    """Parse JSON safely"""
    if not text:
        return fallback
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
    
    return fallback


def _get_color_for_category(category: str) -> str:
    """Get color hex code for category"""
    colors = {
        "Môn học": "#FF6B6B",
        "Địa điểm": "#4ECDC4",
        "Chủ đề": "#45B7D1",
        "Cấp độ": "#FFA07A",
        "Kỹ năng": "#98D8C8",
        "Ngôn ngữ": "#A8E6CF",
    }
    return colors.get(category, "#95A5A6")


def _get_icon_for_category(category: str) -> str:
    """Get icon name for category"""
    icons = {
        "Môn học": "school",
        "Địa điểm": "location_on",
        "Chủ đề": "label",
        "Cấp độ": "trending_up",
        "Kỹ năng": "star",
        "Ngôn ngữ": "language",
    }
    return icons.get(category, "label")


async def suggest_labels_for_note(
    text: str,
    existing_labels: Optional[List[str]] = None,
    account_type: str = "free"
) -> Dict[str, Any]:
    """
    Suggest labels for a note using AI (PRO only feature).
    
    Args:
        text: Note content (will be truncated to first 1000 chars)
        existing_labels: User's existing labels (for consistency)
        account_type: User's account type
    
    Returns:
        Dictionary with suggested labels or upgrade message
    """
    from app.database.models import AccountType
    
    # Check if PRO user
    if account_type == AccountType.FREE.value or account_type == AccountType.FREE:
        return {
            "error": "This feature is only available for PRO users",
            "upgrade_required": True,
            "upgrade_message": "🏷️ AI Label Suggestion chỉ dành cho PRO users. Nâng cấp để tự động phân loại ghi chú!",
            "suggested_labels": [],
            "recommended_categories": []
        }
    
    # Truncate text to first 1000 chars to save tokens
    text_truncated = text[:1000] if text else ""
    
    if not text_truncated.strip():
        return {
            "suggested_labels": [],
            "recommended_categories": [],
            "error": "Empty note content"
        }
    
    # Get AI model for PRO user
    llm = get_chat_llm_for_account(account_type, temperature=0.3)
    
    # Prepare existing labels string
    existing_labels_str = ", ".join(existing_labels) if existing_labels else "Chưa có labels"
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=LABEL_SUGGESTION_PROMPT)
    
    try:
        # Call AI
        print(f"[label_suggester] Suggesting labels for {len(text_truncated)} chars, account_type={account_type}")
        response = await chain.ainvoke({
            "text": text_truncated,
            "existing_labels": existing_labels_str
        })
        
        # Parse response
        result = _safe_json_loads(response.get("text", ""), {})
        
        # Add color and icon to each label
        suggested_labels = result.get("suggested_labels", [])
        for label in suggested_labels:
            category = label.get("category", "")
            label["color"] = _get_color_for_category(category)
            label["icon"] = _get_icon_for_category(category)
        
        # Filter by confidence threshold
        filtered_labels = [
            label for label in suggested_labels 
            if label.get("confidence", 0) >= 0.7
        ]
        
        print(f"[label_suggester] Suggested {len(filtered_labels)} labels (filtered from {len(suggested_labels)})")
        
        return {
            "suggested_labels": filtered_labels,
            "recommended_categories": result.get("recommended_categories", []),
            "is_pro_feature": True
        }
        
    except Exception as e:
        print(f"[label_suggester] Error: {e}")
        return {
            "suggested_labels": [],
            "recommended_categories": [],
            "error": str(e)
        }


def get_popular_labels_by_category() -> Dict[str, List[str]]:
    """
    Get popular labels grouped by category.
    Useful for autocomplete and suggestions.
    """
    return {
        "Môn học": [
            "Toán", "Lý", "Hóa", "Văn", "Anh", "Sử", "Địa", "Sinh", 
            "GDCD", "Tin học", "Công nghệ", "Thể dục"
        ],
        "Địa điểm": [
            "Hà Nội", "TP.HCM", "Đà Nẵng", "Huế", "Hội An", "Nha Trang",
            "Phú Quốc", "Sapa", "Hạ Long", "Việt Nam"
        ],
        "Chủ đề": [
            "Du lịch", "Ẩm thực", "Công nghệ", "Kinh doanh", "Sức khỏe",
            "Thể thao", "Nghệ thuật", "Âm nhạc", "Phim ảnh", "Sách"
        ],
        "Cấp độ": [
            "Tiểu học", "THCS", "THPT", "Đại học", "Cao học",
            "Cơ bản", "Trung cấp", "Nâng cao"
        ],
        "Kỹ năng": [
            "Nghe", "Nói", "Đọc", "Viết", "Ngữ pháp", "Từ vựng",
            "Phát âm", "Giao tiếp", "Dịch thuật"
        ],
        "Ngôn ngữ": [
            "Tiếng Anh", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Trung",
            "Tiếng Pháp", "Tiếng Đức", "Tiếng Tây Ban Nha"
        ]
    }
