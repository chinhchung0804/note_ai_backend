"""
Feature configuration based on account type
"""
from enum import Enum
from typing import List, Dict, Any

class AccountType(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class VocabFeature(str, Enum):
    """Vocab checklist features"""
    SUMMARY_TABLE = "summary_table"
    VOCAB_STORY = "vocab_story"
    VOCAB_MCQS = "vocab_mcqs"
    FLASHCARDS = "flashcards"
    CLOZE_TESTS = "cloze_tests"
    MATCH_PAIRS = "match_pairs"

# Feature configuration by account type
# NOTE: All account types now have access to all 6 vocab features!
# The difference is in daily note limits and AI model quality.
VOCAB_FEATURES_CONFIG = {
    AccountType.FREE: [
        VocabFeature.SUMMARY_TABLE,
        VocabFeature.VOCAB_STORY,
        VocabFeature.VOCAB_MCQS,
        VocabFeature.FLASHCARDS,
        VocabFeature.CLOZE_TESTS,
        VocabFeature.MATCH_PAIRS,
    ],
    AccountType.PRO: [
        VocabFeature.SUMMARY_TABLE,
        VocabFeature.VOCAB_STORY,
        VocabFeature.VOCAB_MCQS,
        VocabFeature.FLASHCARDS,
        VocabFeature.CLOZE_TESTS,
        VocabFeature.MATCH_PAIRS,
    ],
    AccountType.ENTERPRISE: [
        VocabFeature.SUMMARY_TABLE,
        VocabFeature.VOCAB_STORY,
        VocabFeature.VOCAB_MCQS,
        VocabFeature.FLASHCARDS,
        VocabFeature.CLOZE_TESTS,
        VocabFeature.MATCH_PAIRS,
    ],
}

def get_enabled_vocab_features(account_type: str) -> List[str]:
    """
    Get list of enabled vocab features for account type.
    
    Args:
        account_type: "free", "pro", or "enterprise"
    
    Returns:
        List of enabled feature names
    """
    try:
        account_enum = AccountType(account_type.lower())
    except ValueError:
        account_enum = AccountType.FREE
    
    features = VOCAB_FEATURES_CONFIG.get(account_enum, VOCAB_FEATURES_CONFIG[AccountType.FREE])
    return [f.value for f in features]

def is_feature_enabled(account_type: str, feature: str) -> bool:
    """
    Check if a feature is enabled for account type.
    
    Args:
        account_type: "free", "pro", or "enterprise"
        feature: Feature name (e.g., "vocab_story")
    
    Returns:
        True if feature is enabled
    """
    enabled_features = get_enabled_vocab_features(account_type)
    return feature in enabled_features

def get_account_benefits() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed benefits for each account type.
    
    Returns:
        Dictionary with account type as key and benefits as value
    """
    return {
        "free": {
            "name": "FREE",
            "price": "Miễn phí",
            "daily_notes": 3,
            "daily_notes_description": "3 ghi chú/ngày (tất cả loại: text, checklist, file)",
            "ai_model": "GPT-4o-mini",
            "features": {
                "basic_summary": True,
                "questions": True,
                "mcqs": True,
                "vocab_features": 6,  # All 6 features!
                "vocab_story": True,
                "cloze_tests": True,
                "match_pairs": True,
                "ai_label_suggestion": False,  # PRO only
                "priority_support": False,
            },
            "vocab_features_list": [
                "Bảng từ vựng chi tiết",
                "Flashcards SRS",
                "Trắc nghiệm từ vựng",
                "Câu chuyện từ vựng (Vocab Story)",
                "Bài tập điền từ (Cloze Tests)",
                "Trò chơi nối từ (Match Pairs)",
            ],
            "limitations": [
                "Giới hạn 3 ghi chú/ngày (tất cả loại)",
                "AI model: GPT-4o-mini (tốt)",
                "Tự tạo label thủ công",
            ],
            "benefits": [
                "✅ Tất cả 6 tính năng vocab",
                "✅ Xem kết quả học tập đầy đủ",
                "✅ Vocab Story, Cloze Tests, Match Pairs",
                "⚠️ Giới hạn 3 notes/ngày (text + checklist)",
                "⚠️ Tự tạo label thủ công",
            ]
        },
        "pro": {
            "name": "PRO",
            "price": "99,000 VND/tháng",
            "daily_notes": -1,  # Unlimited
            "ai_model": "GPT-4o-mini (có thể nâng GPT-4)",
            "features": {
                "basic_summary": True,
                "questions": True,
                "mcqs": True,
                "vocab_features": 6,  # All features
                "vocab_story": True,
                "cloze_tests": True,
                "match_pairs": True,
                "ai_label_suggestion": True,  # PRO exclusive!
                "priority_support": False,
            },
            "vocab_features_list": [
                "Bảng từ vựng chi tiết",
                "Flashcards SRS",
                "Trắc nghiệm từ vựng",
                "Câu chuyện từ vựng (Vocab Story)",
                "Bài tập điền từ (Cloze Tests)",
                "Trò chơi nối từ (Match Pairs)",
            ],
            "pro_exclusive_features": [
                "🏷️ AI Label Suggestion - Tự động gợi ý labels",
                "🔍 Smart Search - Tìm kiếm thông minh",
                "📊 Learning Analytics - Thống kê học tập",
            ],
            "benefits": [
                "✅ Unlimited ghi chú mỗi ngày",
                "✅ Tất cả 6 tính năng vocab",
                "✅ AI chất lượng cao (GPT-4o-mini)",
                "✅ 🏷️ AI Label Suggestion - Tự động phân loại",
                "✅ Có thể nâng cấp lên GPT-4",
                "✅ Không giới hạn số lượng",
                "✅ Phù hợp cho học tập nghiêm túc",
            ]
        },
        "enterprise": {
            "name": "ENTERPRISE",
            "price": "Liên hệ",
            "daily_notes": -1,  # Unlimited
            "ai_model": "GPT-4 (chất lượng tốt nhất)",
            "features": {
                "basic_summary": True,
                "questions": True,
                "mcqs": True,
                "vocab_features": 6,  # All features
                "vocab_story": True,
                "cloze_tests": True,
                "match_pairs": True,
                "ai_label_suggestion": True,
                "priority_support": True,
            },
            "vocab_features_list": [
                "Bảng từ vựng chi tiết",
                "Flashcards SRS",
                "Trắc nghiệm từ vựng",
                "Câu chuyện từ vựng (Vocab Story)",
                "Bài tập điền từ (Cloze Tests)",
                "Trò chơi nối từ (Match Pairs)",
            ],
            "benefits": [
                "✅ Tất cả tính năng PRO",
                "✅ AI chất lượng cao nhất (GPT-4)",
                "✅ Hỗ trợ ưu tiên",
                "✅ Tùy chỉnh theo nhu cầu",
                "✅ API riêng biệt",
            ]
        }
    }

def get_upgrade_message(feature: str) -> str:
    """
    Get upgrade message for a disabled feature.
    
    Args:
        feature: Feature name (e.g., "vocab_story")
    
    Returns:
        Upgrade message string
    """
    messages = {
        "vocab_story": "📚 Vocab Story chỉ dành cho PRO users. Nâng cấp để học từ vựng qua câu chuyện thú vị!",
        "cloze_tests": "✏️ Cloze Tests chỉ dành cho PRO users. Nâng cấp để luyện tập điền từ hiệu quả!",
        "match_pairs": "🎮 Match Pairs chỉ dành cho PRO users. Nâng cấp để học từ vựng qua trò chơi!",
    }
    return messages.get(feature, f"🌟 Tính năng {feature} chỉ dành cho PRO users. Nâng cấp ngay!")
