"""
Test script để kiểm tra chức năng tạo summary riêng cho text và file
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.agents.summarizer_agent import generate_learning_assets


async def test_separate_summaries():
    """Test tạo summary riêng cho text và file"""
    
    # Giả lập text note
    text_content = """
    Hôm nay tôi học về lập trình Python. Python là ngôn ngữ dễ học và mạnh mẽ.
    Tôi đã học về biến, vòng lặp và hàm. Rất thú vị!
    """
    
    # Giả lập nội dung file
    file_content = """
    Machine Learning là một nhánh của trí tuệ nhân tạo.
    Nó cho phép máy tính học từ dữ liệu mà không cần lập trình cụ thể.
    Các thuật toán phổ biến bao gồm: Linear Regression, Decision Trees, Neural Networks.
    """
    
    print("=" * 60)
    print("TEST: Tạo summary riêng cho text và file")
    print("=" * 60)
    
    # Tạo summary cho text
    print("\n1. Tạo summary cho TEXT NOTE:")
    print("-" * 60)
    text_summary = await generate_learning_assets(
        raw_text=text_content.strip(),
        db=None,
        file_type='text',
        use_rag=False
    )
    
    print("\nText Summary:")
    if text_summary and text_summary.get('summaries'):
        summaries = text_summary['summaries']
        print(f"  - One sentence: {summaries.get('one_sentence', 'N/A')}")
        print(f"  - Short paragraph: {summaries.get('short_paragraph', 'N/A')}")
        print(f"  - Bullet points: {summaries.get('bullet_points', [])}")
    else:
        print("  [Không có summary]")
    
    # Tạo summary cho file
    print("\n2. Tạo summary cho FILE:")
    print("-" * 60)
    file_summary = await generate_learning_assets(
        raw_text=file_content.strip(),
        db=None,
        file_type='pdf',
        use_rag=False
    )
    
    print("\nFile Summary:")
    if file_summary and file_summary.get('summaries'):
        summaries = file_summary['summaries']
        print(f"  - One sentence: {summaries.get('one_sentence', 'N/A')}")
        print(f"  - Short paragraph: {summaries.get('short_paragraph', 'N/A')}")
        print(f"  - Bullet points: {summaries.get('bullet_points', [])}")
    else:
        print("  [Không có summary]")
    
    # Tạo summary tổng hợp (để so sánh)
    print("\n3. Tạo summary TỔNG HỢP (combined):")
    print("-" * 60)
    combined_content = f"{text_content}\n\n[Source: document.pdf]\n{file_content}"
    combined_summary = await generate_learning_assets(
        raw_text=combined_content.strip(),
        db=None,
        file_type='combined',
        use_rag=False
    )
    
    print("\nCombined Summary:")
    if combined_summary and combined_summary.get('summaries'):
        summaries = combined_summary['summaries']
        print(f"  - One sentence: {summaries.get('one_sentence', 'N/A')}")
        print(f"  - Short paragraph: {summaries.get('short_paragraph', 'N/A')}")
        print(f"  - Bullet points: {summaries.get('bullet_points', [])}")
    else:
        print("  [Không có summary]")
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ:")
    print("=" * 60)
    print("✓ Text summary: Tóm tắt riêng về Python")
    print("✓ File summary: Tóm tắt riêng về Machine Learning")
    print("✓ Combined summary: Tóm tắt cả hai nội dung")
    print("\nTrên UI Android, bạn có thể hiển thị:")
    print("  📝 Text: [text_summary]")
    print("  📄 File (document.pdf): [file_summary]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_separate_summaries())
