"""
Prompt Retriever - RAG để cải thiện prompts dựa trên feedback
Sử dụng feedback từ users để cải thiện chất lượng tóm tắt
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.services.feedback_service import feedback_service


class PromptRetriever:
    """
    RAG-based Prompt Retriever
    Sử dụng feedback để cải thiện prompts
    """
    
    @staticmethod
    def get_improved_prompt(
        db: Session,
        base_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Lấy improved prompt dựa trên feedback (RAG)
        
        Args:
            db: Database session
            base_prompt: Base prompt template
            context: Additional context (optional)
            
        Returns:
            Improved prompt với insights từ feedback
        """
        insights = feedback_service.get_improvement_insights(db, limit=5)
        
        improved_prompt = base_prompt
        
        if insights.get('positive_examples'):
            improved_prompt += "\n\n=== Ví dụ tóm tắt tốt (từ feedback tích cực) ===\n"
            for i, example in enumerate(insights['positive_examples'][:3], 1):
                improved_prompt += f"\nVí dụ {i}:\n"
                improved_prompt += f"Input: {example.get('raw_text', '')[:100]}...\n"
                improved_prompt += f"Output: {example.get('summary', '')}\n"
                if example.get('liked_aspects'):
                    improved_prompt += f"Điểm tốt: {', '.join(example['liked_aspects'][:3])}\n"
        
        if insights.get('negative_examples'):
            improved_prompt += "\n\n=== Ví dụ cần tránh (từ feedback tiêu cực) ===\n"
            for i, example in enumerate(insights['negative_examples'][:2], 1):
                improved_prompt += f"\nVí dụ {i} (cần tránh):\n"
                improved_prompt += f"Input: {example.get('raw_text', '')[:100]}...\n"
                improved_prompt += f"Output: {example.get('summary', '')}\n"
                if example.get('disliked_aspects'):
                    improved_prompt += f"Vấn đề: {', '.join(example['disliked_aspects'][:3])}\n"
                if example.get('suggestions'):
                    improved_prompt += f"Gợi ý: {example['suggestions']}\n"
        
        if insights.get('common_liked_aspects'):
            improved_prompt += f"\n\n=== Điểm tốt users thường thích ===\n"
            improved_prompt += f"{', '.join(insights['common_liked_aspects'][:5])}\n"
            improved_prompt += "\nHãy đảm bảo tóm tắt có những điểm này.\n"
        
        if insights.get('common_disliked_aspects'):
            improved_prompt += f"\n\n=== Điểm users thường không thích ===\n"
            improved_prompt += f"{', '.join(insights['common_disliked_aspects'][:5])}\n"
            improved_prompt += "\nHãy tránh những điểm này trong tóm tắt.\n"
        
        if insights.get('suggestions'):
            improved_prompt += f"\n\n=== Gợi ý cải thiện từ users ===\n"
            for suggestion in insights['suggestions'][:3]:
                improved_prompt += f"- {suggestion}\n"
        
        improved_prompt += "\n\n=== Hướng dẫn ===\n"
        improved_prompt += "Dựa trên các ví dụ và feedback ở trên, hãy tạo tóm tắt tốt hơn.\n"
        improved_prompt += "Tóm tắt phải: ngắn gọn (1-3 câu), chính xác, giữ nguyên thông tin quan trọng.\n"
        
        return improved_prompt
    
    @staticmethod
    def get_contextual_prompt(
        db: Session,
        raw_text: str,
        file_type: Optional[str] = None
    ) -> str:
        """
        Lấy contextual prompt dựa trên content type và feedback
        
        Args:
            db: Database session
            raw_text: Raw text cần tóm tắt
            file_type: Type of file (optional)
            
        Returns:
            Contextual prompt
        """
        base_prompt = (
            'Bạn là trợ lý AI giúp tóm tắt nội dung tiếng Việt ngắn gọn, chính xác.\n'
            'Hãy tóm tắt nội dung sau trong 1-3 câu, giữ nguyên thông tin quan trọng:\n\n'
        )
        
        improved_prompt = PromptRetriever.get_improved_prompt(
            db=db,
            base_prompt=base_prompt,
            context={'file_type': file_type}
        )
        
        if file_type:
            if file_type == 'image':
                improved_prompt += "\nLưu ý: Đây là text từ OCR, có thể có lỗi. Hãy sửa lỗi và tóm tắt chính xác.\n"
            elif file_type == 'audio':
                improved_prompt += "\nLưu ý: Đây là text từ STT, có thể có lỗi. Hãy sửa lỗi và tóm tắt chính xác.\n"
            elif file_type == 'pdf' or file_type == 'docx':
                improved_prompt += "\nLưu ý: Đây là text từ document. Hãy tóm tắt nội dung chính.\n"
        
        return improved_prompt
    
    @staticmethod
    def get_simple_prompt() -> str:
        """
        Lấy simple prompt (fallback khi không có feedback)
        """
        return (
            'Bạn là trợ lý AI giúp tóm tắt nội dung tiếng Việt ngắn gọn, chính xác.\n'
            'Hãy tóm tắt nội dung sau trong 1-3 câu, giữ nguyên thông tin quan trọng:\n\n'
        )


prompt_retriever = PromptRetriever()

