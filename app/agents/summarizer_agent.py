import json
import re
import asyncio
from typing import Optional, Dict, Any, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import Session

from app.agents.llm_config import (
    get_openai_chat_llm,
    get_gemini_chat_llm,
)

PRIMARY_LLM = get_openai_chat_llm(temperature=0.3)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "up", "about", "into", "over", "after", "under", "above",
    "below", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "but",
}

summary_prompt_template = PromptTemplate(
    input_variables=['instructions', 'raw_text'],
    template=(
        "{instructions}\n\n"
        "NHIỆM VỤ: Tạo JSON theo đúng schema sau (CHỈ trả về JSON, không có markdown, không có giải thích):\n"
        "{{\n"
        '  "one_sentence": "Tóm tắt 1 câu",\n'
        '  "short_paragraph": "Tóm tắt 3-5 câu",\n'
        '  "bullet_points": ["Ý 1", "Ý 2", "Ý 3"]\n'
        "}}\n"
        "QUAN TRỌNG: Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm.\n\n"
        "NỘI DUNG GỐC:\n{raw_text}\n"
    )
)

question_prompt_template = PromptTemplate(
    input_variables=['raw_text'],
    template=(
        "Bạn là giáo viên chuyên tạo câu hỏi ôn tập chất lượng cao. Dựa vào ghi chú sau, tạo 5-10 câu hỏi tự luận"
        " giúp người học hiểu sâu và nhớ nội dung.\n\n"
        "Yêu cầu:\n"
        "- Mỗi câu hỏi phải kiểm tra hiểu biết, không chỉ nhớ máy móc\n"
        "- Câu hỏi đa dạng: có câu hỏi về khái niệm, so sánh, phân tích, áp dụng\n"
        "- Câu hỏi phải rõ ràng, cụ thể, không mơ hồ\n"
        "- Đáp án phải ngắn gọn nhưng đầy đủ thông tin quan trọng (2-4 câu)\n"
        "- Không được tạo câu hỏi chỉ lặp lại nguyên văn nội dung ghi chú\n\n"
        "Trả về JSON đúng schema (CHỈ JSON, không có markdown):\n"
        "{{\n"
        '  "questions": [\n'
        '    {{"question": "Câu hỏi kiểm tra hiểu biết về khái niệm/ý chính", "answer": "Đáp án ngắn gọn nhưng đầy đủ (2-4 câu)"}},\n'
        '    {{"question": "Câu hỏi yêu cầu so sánh/phân tích", "answer": "Đáp án chi tiết"}},\n'
        '    {{"question": "Câu hỏi về ứng dụng/thực tế", "answer": "Đáp án cụ thể"}}\n'
        "  ]\n"
        "}}\n"
        "QUAN TRỌNG: Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm.\n\n"
        "Ghi chú:\n{raw_text}\n"
    )
)

mcq_prompt_template = PromptTemplate(
    input_variables=['raw_text'],
    template=(
        "Bạn là giáo viên chuyên tạo câu hỏi trắc nghiệm chất lượng cao. Dựa vào ghi chú sau, tạo câu hỏi trắc nghiệm với các yêu cầu:\n\n"
        "YÊU CẦU VỀ CÂU HỎI:\n"
        "- Câu hỏi phải NGẮN GỌN, rõ ràng, chỉ hỏi về một khái niệm/ý chính cụ thể\n"
        "- KHÔNG được copy-paste toàn bộ nội dung ghi chú vào câu hỏi\n"
        "- KHÔNG được đặt câu hỏi dạng 'Dựa trên ghi chú, ý nào mô tả chính xác nhất: [toàn bộ đoạn văn dài]'\n"
        "- Câu hỏi phải độc lập, có thể hiểu được mà không cần đọc lại toàn bộ ghi chú\n"
        "- Mỗi độ khó (easy, medium, hard) tạo từ 1-3 câu hỏi\n"
        "- Mỗi câu hỏi phải kiểm tra hiểu biết về nội dung ghi chú, không chỉ nhớ máy móc\n\n"
        "YÊU CẦU VỀ ĐÁP ÁN:\n"
        "- Mỗi câu có 4 phương án A, B, C, D\n"
        "- Đáp án đúng phải phân bố đều (không phải tất cả đều A)\n"
        "- Các phương án phải NGẮN GỌN, rõ ràng (1-2 câu), không lặp lại toàn bộ nội dung ghi chú\n"
        "- Các phương án sai (distractors) phải:\n"
        "  + Có vẻ hợp lý và liên quan đến chủ đề\n"
        "  + Dựa trên thông tin trong ghi chú nhưng sai hoặc không chính xác\n"
        "  + Không được quá rõ ràng là sai\n"
        "  + Có nội dung thực tế, không được chung chung như 'Nội dung hoàn toàn khác' hoặc 'Không có trong ghi chú'\n"
        "- Explanation phải giải thích ngắn gọn tại sao đáp án đúng và tại sao các phương án khác sai\n\n"
        "VÍ DỤ CÂU HỎI ĐÚNG:\n"
        "- Large Language Model (LLM) là gì?\n"
        "- Kiến trúc chính được sử dụng trong LLM là gì?\n"
        "- Mô hình LLM nào sau đây được phát triển bởi OpenAI?\n\n"
        "VÍ DỤ CÂU HỎI SAI (KHÔNG ĐƯỢC LÀM):\n"
        "- Dựa trên ghi chú, ý nào mô tả chính xác nhất: [copy toàn bộ đoạn văn dài từ ghi chú]\n"
        "- Theo nội dung trên, thông tin nào đúng: [paste lại toàn bộ định nghĩa]\n\n"
        "Trả về JSON đúng schema (CHỈ JSON, không có markdown):\n"
        "{{\n"
        '  "easy": [\n'
        '    {{"question": "Câu hỏi ngắn gọn về khái niệm cơ bản (1 câu)", "options": {{"A": "Đáp án đúng ngắn gọn (1-2 câu)", "B": "Phương án sai nhưng hợp lý (1-2 câu)", "C": "Phương án sai nhưng liên quan (1-2 câu)", "D": "Phương án sai nhưng có vẻ đúng (1-2 câu)"}}, "answer": "A", "explanation": "Giải thích ngắn gọn tại sao A đúng và B, C, D sai"}}\n'
        "  ],\n"
        '  "medium": [\n'
        '    {{"question": "Câu hỏi ngắn gọn yêu cầu hiểu sâu hơn (1 câu)", "options": {{"A": "Phương án sai ngắn gọn", "B": "Đáp án đúng ngắn gọn", "C": "Phương án sai ngắn gọn", "D": "Phương án sai ngắn gọn"}}, "answer": "B", "explanation": "Giải thích ngắn gọn"}}\n'
        "  ],\n"
        '  "hard": [\n'
        '    {{"question": "Câu hỏi ngắn gọn về phân tích, so sánh hoặc áp dụng (1 câu)", "options": {{"A": "Phương án sai ngắn gọn", "B": "Phương án sai ngắn gọn", "C": "Đáp án đúng ngắn gọn", "D": "Phương án sai ngắn gọn"}}, "answer": "C", "explanation": "Giải thích ngắn gọn"}}\n'
        "  ]\n"
        "}}\n"
        "QUAN TRỌNG:\n"
        "- Trả về CHỈ JSON thuần túy, không có ```json``` hoặc text thêm\n"
        "- Đảm bảo đáp án đúng phân bố đều giữa A, B, C, D\n"
        "- Câu hỏi và đáp án phải NGẮN GỌN, KHÔNG lặp lại toàn bộ nội dung ghi chú\n"
        "- Câu hỏi phải độc lập, có thể hiểu được mà không cần đọc lại ghi chú\n\n"
        "Ghi chú:\n{raw_text}\n"
    )
)

summary_chain = LLMChain(llm=PRIMARY_LLM, prompt=summary_prompt_template)
question_chain = LLMChain(llm=PRIMARY_LLM, prompt=question_prompt_template)
mcq_chain = LLMChain(llm=PRIMARY_LLM, prompt=mcq_prompt_template)

vocab_summary_table_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là chuyên gia dạy từ vựng tiếng Anh. Dựa trên danh sách từ vựng và ngữ cảnh sau, "
        "hãy tạo bảng tóm tắt CHI TIẾT và THỰC TẾ cho từng từ.\n\n"
        "FORMAT YÊU CẦU:\n"
        "- Trả về CHỈ JSON thuần túy (mảng), không có markdown code blocks (```json```)\n"
        "- Không có text thêm trước hoặc sau JSON\n"
        "- Đảm bảo tất cả Unicode characters được encode đúng (không có invalid \\u escape sequences)\n"
        "- Sử dụng double quotes cho strings, escape đúng các ký tự đặc biệt\n\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Tối đa 15 từ vựng.\n"
        "- Nếu vocab_list có từ, ƯU TIÊN dùng các từ đó; nếu không, tự chọn từ khóa chính trong raw_text.\n"
        "- Loại bỏ stopwords (the, a, an, and, with, of, ...), chỉ giữ danh/động/tính từ nội dung.\n"
        "- Mỗi từ PHẢI có nội dung THỰC TẾ, CỤ THỂ, KHÔNG dùng placeholder như 'Nghĩa của X', 'Định nghĩa ngắn gọn về X'.\n"
        "- Translation: nghĩa tiếng Việt CHÍNH XÁC của từ\n"
        "- Definition: định nghĩa tiếng Anh ngắn gọn nhưng ĐẦY ĐỦ (1-2 câu)\n"
        "- Usage_note: hướng dẫn cách dùng CỤ THỂ trong ngữ cảnh thực tế\n"
        "- Common_structures: các cấu trúc ngữ pháp THỰC TẾ mà từ này thường dùng (ví dụ: 'to + verb', 'verb + object')\n"
        "- Collocations: các cụm từ THỰC TẾ thường đi kèm với từ này (ví dụ: 'make a decision', 'take action')\n\n"
        "VÍ DỤ ĐÚNG:\n"
        "{{\n"
        '  "word": "decision",\n'
        '  "translation": "quyết định",\n'
        '  "part_of_speech": "noun",\n'
        '  "definition": "a choice or judgment that you make after thinking about various possibilities",\n'
        '  "usage_note": "Dùng với động từ make/take: make a decision, take a decision. Không dùng do a decision",\n'
        '  "common_structures": ["make a decision", "decision to do something", "decision on/about"],\n'
        '  "collocations": ["make a decision", "reach a decision", "final decision", "tough decision"]\n'
        "}}\n\n"
        "VÍ DỤ SAI (KHÔNG ĐƯỢC LÀM):\n"
        "{{\n"
        '  "word": "decision",\n'
        '  "translation": "Nghĩa của decision",\n'
        '  "definition": "Định nghĩa ngắn gọn về decision",\n'
        '  "usage_note": "Cách dùng decision trong câu"\n'
        "}}\n\n"
        "Schema JSON:\n"
        "[\n"
        '  {{\n'
        '    "word": "từ vựng thực tế",\n'
        '    "translation": "nghĩa tiếng Việt chính xác",\n'
        '    "part_of_speech": "noun/verb/adj/adv",\n'
        '    "definition": "định nghĩa tiếng Anh đầy đủ (1-2 câu)",\n'
        '    "usage_note": "hướng dẫn cách dùng cụ thể",\n'
        '    "common_structures": ["cấu trúc thực tế 1", "cấu trúc thực tế 2"],\n'
        '    "collocations": ["cụm từ thực tế 1", "cụm từ thực tế 2"]\n'
        '  }}\n'
        "]\n\n"
        "vocab_list:\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

vocab_story_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là người viết truyện ngắn sáng tạo để học từ vựng tiếng Anh. Dựa trên vocab_list và raw_text, "
        "hãy viết một câu chuyện DÀI HƠN, HẤP DẪN, có cốt truyện HOÀN CHỈNH với ngữ cảnh QUEN THUỘC và DỄ HIỂU.\n\n"
        "FORMAT YÊU CẦU:\n"
        "- Trả về CHỈ JSON thuần túy, không có markdown code blocks (```json```)\n"
        "- Không có text thêm trước hoặc sau JSON\n"
        "- Đảm bảo tất cả Unicode characters được encode đúng (không có invalid \\u escape sequences)\n"
        "- Sử dụng double quotes cho strings, escape đúng các ký tự đặc biệt\n\n"
        "YÊU CẦU QUAN TRỌNG (BẮT BUỘC):\n"
        "- Câu chuyện phải TỰ NHIÊN, có cốt truyện RÕ RÀNG và HOÀN CHỈNH với mở đầu, phát triển, và kết thúc\n"
        "- ĐỘ DÀI BẮT BUỘC: Tối thiểu 5 đoạn, mỗi đoạn tối thiểu 3 câu (tổng cộng ít nhất 15 câu). KHÔNG được tạo câu chuyện ngắn chỉ 2-3 câu!\n"
        "- Tất cả các từ trong vocab_list PHẢI xuất hiện trong câu chuyện và được đánh dấu **bold** trong paragraphs; KHÔNG đưa stopwords như 'the/and/with'\n"
        "- Mỗi từ trong vocab_list nên xuất hiện ÍT NHẤT 1-2 lần trong câu chuyện để người học có cơ hội gặp lại\n"
        "- Ngữ cảnh phải QUEN THUỘC, dễ liên tưởng (ví dụ: cuộc sống hàng ngày, học tập, công việc, du lịch, tình bạn)\n"
        "- Câu chuyện phải có TÍNH LIÊN KẾT giữa các đoạn, tạo thành một câu chuyện HOÀN CHỈNH, không phải các câu rời rạc\n"
        "- Câu chuyện phải giúp người học GHI NHỚ từ vựng thông qua ngữ cảnh và cốt truyện hấp dẫn\n"
        "- Sử dụng từ vựng một cách TỰ NHIÊN, đúng ngữ pháp, trong các tình huống thực tế\n"
        "- Nếu vocab_list có nhiều từ, hãy phân bố đều các từ trong suốt câu chuyện, không tập trung tất cả ở đầu hoặc cuối\n"
        "- LƯU Ý: Nếu bạn tạo câu chuyện quá ngắn (dưới 5 đoạn), kết quả sẽ bị từ chối. Hãy viết một câu chuyện ĐẦY ĐỦ và HẤP DẪN!\n\n"
        "VÍ DỤ:\n"
        "vocab_list: decision, option, choice, reduction, provider\n"
        "Câu chuyện:\n"
        "{{\n"
        '  "title": "A Difficult Decision",\n'
        '  "paragraphs": [\n'
        '    "Sarah stood at the crossroads, facing a difficult **decision**. She had two job offers: one in the city, one in the countryside.",\n'
        '    "Each **option** had its advantages. The city job offered more money, but the countryside **choice** promised a better work-life balance. She consulted her career **provider** for advice.",\n'
        '    "The career **provider** suggested considering a **reduction** in salary might be worth it for better quality of life. This made Sarah reconsider her **decision** carefully.",\n'
        '    "After weighing all the **options**, Sarah realized that the **choice** wasn\'t just about money. A small **reduction** in income could lead to greater happiness.",\n'
        '    "Finally, she made her **decision** and chose the countryside position. Her career **provider** congratulated her on making the right **choice**."\n'
        '  ],\n'
        '  "used_words": [{{"word": "decision", "bolded": true}}, {{"word": "option", "bolded": true}}, {{"word": "choice", "bolded": true}}, {{"word": "reduction", "bolded": true}}, {{"word": "provider", "bolded": true}}]\n'
        "}}\n\n"
        "Schema JSON:\n"
        "{{\n"
        '  "title": "Tiêu đề câu chuyện ngắn gọn",\n'
        '  "paragraphs": ["Đoạn 1 với **từ in đậm**", "Đoạn 2 với **từ in đậm**", "... (5-8 đoạn)"],\n'
        '  "used_words": [{{"word": "từ 1", "bolded": true}}, {{"word": "từ 2", "bolded": true}}]\n'
        "}}\n\n"
        "QUAN TRỌNG:\n"
        "- vocab_list là DANH SÁCH CÁC TỪ RIÊNG BIỆT, mỗi từ trên một dòng hoặc cách nhau bởi dấu phẩy\n"
        "- Tất cả các từ trong vocab_list PHẢI xuất hiện trong paragraphs và được đánh dấu **bold**\n"
        "- Mỗi từ trong vocab_list là MỘT TỪ RIÊNG BIỆT, không phải một cụm từ dài\n"
        "- Câu chuyện phải DÀI HƠN (5-8 đoạn, mỗi đoạn 3-5 câu) để tạo ngữ cảnh đầy đủ cho việc học từ vựng\n"
        "- Câu chuyện phải có cốt truyện HOÀN CHỈNH, không phải các câu rời rạc\n"
        "- KHÔNG được coi toàn bộ vocab_list như một từ duy nhất; phải tách từng từ và sử dụng riêng biệt\n\n"
        "vocab_list (mỗi từ trên một dòng):\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

vocab_mcq_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là giáo viên chuyên tạo câu hỏi trắc nghiệm từ vựng chất lượng cao. Tạo 8-10 câu MCQ, "
        "mỗi câu có 4 đáp án A-D, có lời giải thích chi tiết. Câu hỏi bằng tiếng Việt, đáp án là từ vựng tiếng Anh trong vocab_list.\n\n"
        "FORMAT YÊU CẦU:\n"
        "- Trả về CHỈ JSON thuần túy (mảng), không có markdown code blocks (```json```)\n"
        "- Không có text thêm trước hoặc sau JSON\n"
        "- Đảm bảo tất cả Unicode characters được encode đúng (không có invalid \\u escape sequences)\n"
        "- Sử dụng double quotes cho strings, escape đúng các ký tự đặc biệt\n\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Mỗi câu hỏi phải kiểm tra hiểu biết về TỪ VỰNG CỤ THỂ, không phải generic\n"
        "- Câu hỏi phải RÕ RÀNG, CỤ THỂ về nghĩa, cách dùng, hoặc ngữ cảnh của từ\n"
        "- Các đáp án PHẢI có NỘI DUNG THỰC TẾ, CỤ THỂ, KHÔNG được dùng placeholder như 'Ý đúng về X' hay 'Ý sai gần với X'\n"
        "- Đáp án đúng phải là định nghĩa/nghĩa/cách dùng CHÍNH XÁC của từ\n"
        "- Các đáp án sai phải là định nghĩa/nghĩa/cách dùng của từ KHÁC nhưng có vẻ hợp lý\n"
        "- Explanation phải giải thích CỤ THỂ tại sao đáp án đúng và tại sao các phương án khác sai\n"
        "- When_wrong phải đưa ra gợi ý CỤ THỂ để nhớ từ này\n\n"
        "- Chỉ dùng từ vựng nội dung (không dùng stopwords như the/and/with)\n"
        "CÁC LOẠI CÂU HỎI:\n"
        "1. Câu hỏi về nghĩa: 'Từ nào sau đây có nghĩa là [định nghĩa cụ thể]?'\n"
        "2. Câu hỏi điền từ: 'Chọn từ phù hợp: I need to [context cụ thể]'\n"
        "3. Câu hỏi về cách dùng: 'Từ nào được dùng đúng trong câu: [câu ví dụ]?'\n"
        "4. Câu hỏi về collocation: 'Từ nào đi với [từ khác] để tạo cụm từ đúng?'\n\n"
        "VÍ DỤ ĐÚNG:\n"
        "{{\n"
        '  "id": 1,\n'
        '  "type": "vocab_mcq",\n'
        '  "vocab_target": "decision",\n'
        '  "question": "Từ nào sau đây có nghĩa là \'a choice or judgment made after thinking\'?",\n'
        '  "options": {{\n'
        '    "A": "decision",\n'
        '    "B": "option",\n'
        '    "C": "selection",\n'
        '    "D": "preference"\n'
        '  }},\n'
        '  "answer": "A",\n'
        '  "explanation": "Decision có nghĩa là quyết định sau khi suy nghĩ. Option là lựa chọn, selection là sự lựa chọn, preference là sở thích",\n'
        '  "when_wrong": "Nhớ: decision = quyết định (make a decision), khác với option (lựa chọn) hay preference (sở thích)"\n'
        "}}\n\n"
        "VÍ DỤ SAI (KHÔNG ĐƯỢC LÀM):\n"
        "{{\n"
        '  "question": "Từ nào diễn đạt ý gần nhất với \'decision\'?",\n'
        '  "options": {{\n'
        '    "A": "Ý đúng về decision",\n'
        '    "B": "Ý sai gần với decision",\n'
        '    "C": "Ý sai khác về decision",\n'
        '    "D": "Ý sai không liên quan decision"\n'
        '  }}\n'
        "}}\n\n"
        "Schema JSON (mảng):\n"
        "[\n"
        '  {{\n'
        '    "id": 1,\n'
        '    "type": "vocab_mcq",\n'
        '    "vocab_target": "từ đang kiểm tra",\n'
        '    "question": "Câu hỏi cụ thể về nghĩa/cách dùng/ngữ cảnh",\n'
        '    "options": {{"A": "đáp án cụ thể 1", "B": "đáp án cụ thể 2", "C": "đáp án cụ thể 3", "D": "đáp án cụ thể 4"}},\n'
        '    "answer": "A",\n'
        '    "explanation": "Giải thích cụ thể tại sao đúng/sai",\n'
        '    "when_wrong": "Gợi ý cụ thể để nhớ từ này"\n'
        '  }}\n'
        "]\n\n"
        "vocab_list:\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

flashcards_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là chuyên gia tạo flashcards SRS cho từ vựng tiếng Anh. Tạo danh sách thẻ CHI TIẾT và THỰC TẾ.\n\n"
        "FORMAT YÊU CẦU:\n"
        "- Trả về CHỈ JSON thuần túy (mảng), không có markdown code blocks (```json```)\n"
        "- Không có text thêm trước hoặc sau JSON\n"
        "- Đảm bảo tất cả Unicode characters được encode đúng (không có invalid \\u escape sequences)\n"
        "- Sử dụng double quotes cho strings, escape đúng các ký tự đặc biệt\n\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Mỗi flashcard PHẢI có nội dung THỰC TẾ, CỤ THỂ, KHÔNG được dùng placeholder\n"
        "- Không dùng stopwords (the/and/with...).\n"
        "- Meaning: nghĩa tiếng Việt CHÍNH XÁC và đầy đủ của từ\n"
        "- Example: câu ví dụ THỰC TẾ, đúng ngữ pháp, có ngữ cảnh rõ ràng\n"
        "- Usage_note: hướng dẫn cách dùng CỤ THỂ, lưu ý quan trọng khi dùng từ này\n"
        "- Synonyms: danh sách từ đồng nghĩa THỰC TẾ (nếu có)\n"
        "- Antonyms: danh sách từ trái nghĩa THỰC TẾ (nếu có)\n"
        "- Recall_task: nhiệm vụ gợi nhớ CỤ THỂ để học từ này\n\n"
        "VÍ DỤ ĐÚNG:\n"
        "{{\n"
        '  "word": "decision",\n'
        '  "front": "decision",\n'
        '  "back": {{\n'
        '    "meaning": "quyết định, sự lựa chọn sau khi suy nghĩ kỹ",\n'
        '    "example": "I made a difficult decision to change my career path",\n'
        '    "usage_note": "Thường dùng với make/take: make a decision. Không dùng do a decision. Decision thường đi với about/on",\n'
        '    "synonyms": ["choice", "judgment", "resolution"],\n'
        '    "antonyms": ["indecision", "hesitation"]\n'
        '  }},\n'
        '  "srs_schedule": {{\n'
        '    "intervals": [1, 3, 7, 14],\n'
        '    "recall_task": "Nhắc lại nghĩa và đặt câu với make a decision"\n'
        '  }}\n'
        "}}\n\n"
        "VÍ DỤ SAI (KHÔNG ĐƯỢC LÀM):\n"
        "{{\n"
        '  "word": "decision",\n'
        '  "back": {{\n'
        '    "meaning": "Nghĩa ngắn gọn của decision",\n'
        '    "example": "Ví dụ: I often practice decision every day",\n'
        '    "usage_note": "Cách dùng decision trong câu",\n'
        '    "synonyms": ["decision_syn1", "decision_syn2"],\n'
        '    "antonyms": ["decision_ant1"]\n'
        '  }}\n'
        "}}\n\n"
        "Schema JSON (mảng):\n"
        "[\n"
        '  {{\n'
        '    "word": "từ vựng",\n'
        '    "front": "từ cần học",\n'
        '    "back": {{\n'
        '      "meaning": "nghĩa tiếng Việt chính xác và đầy đủ",\n'
        '      "example": "câu ví dụ thực tế, đúng ngữ pháp",\n'
        '      "usage_note": "hướng dẫn cách dùng cụ thể",\n'
        '      "synonyms": ["từ đồng nghĩa thực tế"],\n'
        '      "antonyms": ["từ trái nghĩa thực tế"]\n'
        '    }},\n'
        '    "srs_schedule": {{\n'
        '      "intervals": [1, 3, 7, 14],\n'
        '      "recall_task": "nhiệm vụ gợi nhớ cụ thể"\n'
        '    }}\n'
        '  }}\n'
        "]\n\n"
        "vocab_list ưu tiên, nếu trống thì trích từ raw_text.\n\n"
        "vocab_list:\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

mindmap_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là AI chuyên tạo mindmap nhóm từ vựng tiếng Anh. Phân loại và nhóm các từ vựng một cách LOGIC và HỮU ÍCH.\n\n"
        "FORMAT YÊU CẦU:\n"
        "- Trả về CHỈ JSON thuần túy, không có markdown code blocks (```json```)\n"
        "- Không có text thêm trước hoặc sau JSON\n"
        "- Đảm bảo tất cả Unicode characters được encode đúng (không có invalid \\u escape sequences)\n"
        "- Sử dụng double quotes cho strings, escape đúng các ký tự đặc biệt\n\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Loại bỏ stopwords (the/and/with...). Chỉ dùng danh/động/tính từ nội dung.\n"
        "- by_topic: Nhóm từ theo CHỦ ĐỀ CỤ THỂ (ví dụ: 'Animals', 'Food', 'Technology'), KHÔNG dùng 'General'.\n"
        "- by_difficulty: Phân loại theo MỨC ĐỘ THỰC TẾ (easy = từ cơ bản, medium = từ trung bình, hard = từ nâng cao)\n"
        "- by_pos: Nhóm theo LOẠI TỪ (noun, verb, adjective, adverb)\n"
        "- by_relation: Nhóm theo QUAN HỆ (synonyms/antonyms/related), KHÔNG dùng 'related_terms' chung chung.\n"
        "- Description phải CỤ THỂ, giải thích tại sao nhóm này lại có các từ này; tránh mô tả chung chung.\n"
        "- Ưu tiên dùng vocab_list nếu có, nếu không hãy chọn từ khóa chính trong raw_text\n\n"
        "VÍ DỤ:\n"
        "vocab_list: fish, whale, dolphin, decision, choice, option\n"
        "{{\n"
        '  "by_topic": [\n'
        '    {{"topic": "Marine Animals", "description": "Các loài động vật biển", "words": ["fish", "whale", "dolphin"]}},\n'
        '    {{"topic": "Decision Making", "description": "Từ vựng về quyết định và lựa chọn", "words": ["decision", "choice", "option"]}}\n'
        '  ],\n'
        '  "by_difficulty": [\n'
        '    {{"level": "easy", "description": "Từ cơ bản, thường gặp", "words": ["fish", "choice"]}},\n'
        '    {{"level": "medium", "description": "Từ trung bình, cần hiểu ngữ cảnh", "words": ["whale", "decision", "option"]}},\n'
        '    {{"level": "hard", "description": "Từ nâng cao, ít gặp", "words": ["dolphin"]}}\n'
        '  ],\n'
        '  "by_pos": [\n'
        '    {{"pos": "noun", "words": ["fish", "whale", "dolphin", "decision", "choice", "option"]}}\n'
        '  ],\n'
        '  "by_relation": [\n'
        '    {{"group_name": "synonyms", "description": "Từ đồng nghĩa về quyết định", "words": ["decision", "choice", "option"], "clusters": [["decision", "choice"], ["option", "choice"]]}},\n'
        '    {{"group_name": "related", "description": "Từ liên quan đến động vật biển", "words": ["fish", "whale", "dolphin"]}}\n'
        '  ]\n'
        "}}\n\n"
        "Schema JSON:\n"
        "{{\n"
        '  "by_topic": [{{"topic": "chủ đề cụ thể", "description": "mô tả cụ thể", "words": ["từ1","từ2"]}}],\n'
        '  "by_difficulty": [{{"level": "easy/medium/hard", "description": "mô tả mức độ", "words": ["..."]}}],\n'
        '  "by_pos": [{{"pos": "noun/verb/adj/adv", "words": ["..."]}}],\n'
        '  "by_relation": [{{"group_name": "synonyms/antonyms/related", "description": "mô tả quan hệ", "words": ["..."], "clusters": [["từ1", "từ2"]], "pairs": [["từ1", "từ2"]]}}]\n'
        "}}\n\n"
        "vocab_list:\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

cloze_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là AI tạo bài Cloze Test để luyện từ vựng. Dựa trên vocab_list và raw_text, hãy tạo NHIỀU câu hỏi ĐA DẠNG "
        "(mỗi câu hỏi là một câu văn độc lập với MỘT chỗ trống duy nhất). "
        "Mỗi chỗ trống là MỘT từ vựng trong vocab_list. "
        "Mỗi câu hỏi chỉ có MỘT chỗ trống để dễ điền.\n\n"
        "FORMAT YÊU CẦU (CỰC KỲ QUAN TRỌNG):\n"
        "- Trả về CHỈ JSON thuần (mảng), KHÔNG có markdown code blocks (```json``` hoặc ```)\n"
        "- KHÔNG có text thêm trước hoặc sau JSON (không có giải thích, không có câu nói thêm)\n"
        "- Bắt đầu response bằng ký tự [ (không có text trước)\n"
        "- Kết thúc response bằng ký tự ] (không có text sau)\n"
        "- Mỗi phần tử là một cloze item với trường rõ ràng.\n"
        "- Mỗi cloze item là một câu hỏi riêng biệt với MỘT chỗ trống duy nhất.\n"
        "- KHÔNG được thêm bất kỳ text nào ngoài JSON, kể cả câu giải thích hay chú thích\n\n"
        "YÊU CẦU NỘI DUNG QUAN TRỌNG (BẮT BUỘC):\n"
        "- BẮT BUỘC: Tất cả các từ trong vocab_list PHẢI được sử dụng làm đáp án cho các chỗ trống (mỗi từ một câu hỏi riêng)\n"
        "- BẮT BUỘC: Mỗi cloze item chỉ có MỘT chỗ trống duy nhất (dạng ___1___), KHÔNG được tạo nhiều chỗ trống trong một paragraph\n"
        "- Loại bỏ stopwords (the/and/with...), chỉ dùng từ vựng nội dung từ vocab_list.\n"
        "- Mỗi câu hỏi phải ĐỘC LẬP, có MỘT chỗ trống duy nhất, dễ hiểu và dễ điền\n"
        "- Câu văn phải TỰ NHIÊN, đa dạng về ngữ cảnh (học tập, công việc, đời sống, khoa học, nghệ thuật, v.v.)\n"
        "- Đa dạng về loại câu hỏi:\n"
        "  + Câu hỏi về nghĩa: \"The word that means 'giảm' is ___.\"\n"
        "  + Câu điền từ vào ngữ cảnh: \"I need to make a ___ about my future career.\"\n"
        "  + Câu về collocation: \"We should ___ the price to attract more customers.\"\n"
        "  + Câu về cách dùng: \"The company decided to ___ its workforce by 10%.\"\n"
        "- Cung cấp đáp án cho từng blank, giải thích ngắn gọn tại sao đúng, và ví dụ mới khi người học trả lời đúng.\n"
        "- Không lặp lại nguyên văn raw_text; viết lại súc tích, dễ hiểu, với ngữ cảnh rõ ràng.\n"
        "- Phân bố đều các từ trong vocab_list, không tập trung một vài từ.\n"
        "- Mỗi câu hỏi phải có ngữ cảnh đủ để người học có thể đoán được từ cần điền.\n"
        "- LƯU Ý: Nếu vocab_list có 8 từ, bạn PHẢI tạo 8 câu hỏi riêng biệt, mỗi câu 1 blank, không được gộp nhiều từ vào một paragraph!\n\n"
        "VÍ DỤ:\n"
        "vocab_list: decision, reduction, provider, option\n"
        "[\n"
        "  {{\n"
        '    "title": "Making Choices",\n'
        '    "paragraph": "Sarah had to make a difficult ___1___ about her career path.",\n'
        '    "blanks": [\n'
        '      {{"id": 1, "answer": "decision", "explanation": "Decision means a choice made after thinking carefully.", "on_correct_example": "I made a quick decision to accept the job offer."}}\n'
        '    ]\n'
        "  }},\n"
        "  {{\n"
        '    "title": "Business Strategy",\n'
        '    "paragraph": "The company announced a 20% ___2___ in staff to reduce costs.",\n'
        '    "blanks": [\n'
        '      {{"id": 2, "answer": "reduction", "explanation": "Reduction means making something smaller or less.", "on_correct_example": "There was a significant reduction in pollution levels."}}\n'
        '    ]\n'
        "  }},\n"
        "  {{\n"
        '    "title": "Service Options",\n'
        '    "paragraph": "We need to find a reliable internet ___3___ for our office.",\n'
        '    "blanks": [\n'
        '      {{"id": 3, "answer": "provider", "explanation": "Provider means a person or company that supplies a service.", "on_correct_example": "The healthcare provider offered excellent service."}}\n'
        '    ]\n'
        "  }},\n"
        "  {{\n"
        '    "title": "Available Choices",\n'
        '    "paragraph": "You have three ___4___: stay, leave, or negotiate.",\n'
        '    "blanks": [\n'
        '      {{"id": 4, "answer": "option", "explanation": "Option means a choice or possibility.", "on_correct_example": "Studying abroad is a great option for students."}}\n'
        '    ]\n'
        "  }}\n"
        "]\n\n"
        "Schema JSON (mảng):\n"
        "[\n"
        "  {{\n"
        '    "title": "tiêu đề ngắn gọn cho câu hỏi",\n'
        '    "paragraph": "Câu văn có MỘT chỗ trống duy nhất dạng ___1___",\n'
        '    "blanks": [\n'
        '      {{"id": 1, "answer": "từ_vựng_từ_vocab_list", "explanation": "giải thích ngắn gọn tại sao đúng", "on_correct_example": "câu ví dụ mới dùng đúng từ này"}}\n'
        '    ]\n'
        "  }},\n"
        "  ... (tạo số lượng câu hỏi BẰNG số lượng từ trong vocab_list, mỗi câu 1 chỗ trống, mỗi từ một câu hỏi riêng)\n"
        "]\n\n"
        "QUAN TRỌNG:\n"
        "- vocab_list là DANH SÁCH CÁC TỪ RIÊNG BIỆT, mỗi từ trên một dòng hoặc cách nhau bởi dấu phẩy\n"
        "- Mỗi câu hỏi chỉ có MỘT chỗ trống để dễ điền\n"
        "- Tất cả các từ trong vocab_list PHẢI được sử dụng làm đáp án (mỗi từ là một đáp án riêng biệt)\n"
        "- Mỗi từ trong vocab_list là MỘT TỪ RIÊNG BIỆT, không phải một cụm từ dài\n"
        "- Câu hỏi phải ĐA DẠNG về ngữ cảnh và loại câu hỏi\n"
        "- Mỗi câu hỏi phải ĐỘC LẬP và có ngữ cảnh đủ để đoán được từ cần điền\n"
        "- KHÔNG được coi toàn bộ vocab_list như một từ duy nhất; phải tách từng từ và tạo câu hỏi riêng cho mỗi từ\n\n"
        "vocab_list (mỗi từ trên một dòng):\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

match_pairs_template = PromptTemplate(
    input_variables=['raw_text', 'vocab_list'],
    template=(
        "Bạn là AI tạo trò chơi nối từ - nghĩa (4x4 gồm 8 cặp). Chọn 8 từ vựng tiêu biểu từ vocab_list (hoặc từ khóa của raw_text nếu thiếu) "
        "và tạo cặp word-meaning rõ ràng.\n\n"
        "FORMAT YÊU CẦU (CỰC KỲ QUAN TRỌNG):\n"
        "- Trả về CHỈ JSON thuần (mảng), KHÔNG có markdown code blocks (```json``` hoặc ```)\n"
        "- KHÔNG có text thêm trước hoặc sau JSON (không có giải thích, không có câu nói thêm)\n"
        "- Bắt đầu response bằng ký tự [ (không có text trước)\n"
        "- Kết thúc response bằng ký tự ] (không có text sau)\n"
        "- Mỗi phần tử là một cặp từ - nghĩa, kèm gợi ý ngắn.\n"
        "- KHÔNG được dùng placeholder như 'Ý nghĩa ngắn gọn, thực tế của X' hoặc 'Nghĩa của X'\n"
        "- Nghĩa phải là NGHĨA THỰC TẾ, CỤ THỂ của từ đó bằng tiếng Việt\n"
        "- KHÔNG được thêm bất kỳ text nào ngoài JSON, kể cả câu giải thích hay chú thích\n\n"
        "VÍ DỤ ĐÚNG:\n"
        "vocab_list: decision, reduction, provider\n"
        "[\n"
        '  {{"id": 1, "word": "decision", "meaning": "quyết định", "hint": "hành động chọn lựa"}},\n'
        '  {{"id": 2, "word": "reduction", "meaning": "sự giảm bớt", "hint": "làm nhỏ hơn"}},\n'
        '  {{"id": 3, "word": "provider", "meaning": "nhà cung cấp", "hint": "người/đơn vị cung cấp dịch vụ"}}\n'
        "]\n\n"
        "VÍ DỤ SAI (KHÔNG ĐƯỢC LÀM):\n"
        "[\n"
        '  {{"id": 1, "word": "decision", "meaning": "Ý nghĩa ngắn gọn, thực tế của decision", "hint": "..."}},\n'
        '  {{"id": 2, "word": "reduction", "meaning": "Nghĩa của reduction", "hint": "..."}}\n'
        "]\n\n"
        "Schema JSON (mảng):\n"
        "[\n"
        '  {{"id": 1, "word": "từ 1", "meaning": "nghĩa tiếng Việt thực tế (ví dụ: quyết định, giảm bớt, nhà cung cấp)", "hint": "gợi ý ngắn (nếu có)"}},\n'
        '  {{"id": 2, "word": "từ 2", "meaning": "nghĩa thực tế khác", "hint": "gợi ý ngắn"}},\n'
        '  ... đến 8 cặp\n'
        "]\n\n"
        "YÊU CẦU NỘI DUNG (BẮT BUỘC):\n"
        "- Nghĩa phải là NGHĨA THỰC TẾ, CỤ THỂ của từ bằng tiếng Việt (ví dụ: 'quyết định', 'giảm bớt', 'nhà cung cấp', 'sò', 'cá voi')\n"
        "- KHÔNG được dùng placeholder như 'Ý nghĩa của X', 'Nghĩa ngắn gọn của X', 'Meaning of X', 'Ý nghĩa ngắn gọn, thực tế của X'\n"
        "- Nghĩa phải là TỪ/CỤM TỪ NGẮN GỌN (1-3 từ), không phải câu dài\n"
        "- Ví dụ ĐÚNG: 'word': 'snail', 'meaning': 'sò' hoặc 'word': 'whale', 'meaning': 'cá voi'\n"
        "- Ví dụ SAI: 'word': 'snail', 'meaning': 'Ý nghĩa ngắn gọn, thực tế của snail'\n"
        "- Hint ngắn gọn (tùy chọn) giúp nhớ nhanh\n"
        "- Không lặp từ; ưu tiên đa dạng loại từ/chủ đề nếu có\n"
        "- Nếu vocab_list có ít hơn 8 từ, chọn tất cả; nếu nhiều hơn, chọn 8 từ tiêu biểu nhất\n\n"
        "vocab_list:\n{vocab_list}\n\n"
        "raw_text:\n{raw_text}\n"
    )
)

vocab_summary_table_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_summary_table_template)
vocab_story_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_story_template)
vocab_mcq_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_mcq_template)
flashcards_chain = LLMChain(llm=PRIMARY_LLM, prompt=flashcards_template)
mindmap_chain = LLMChain(llm=PRIMARY_LLM, prompt=mindmap_template)
cloze_chain = LLMChain(llm=PRIMARY_LLM, prompt=cloze_template)
match_pairs_chain = LLMChain(llm=PRIMARY_LLM, prompt=match_pairs_template)

_fallback_chains: Dict[str, Optional[LLMChain]] = {
    'summary': None,
    'question': None,
    'mcq': None,
    'vocab_summary_table': None,
    'vocab_story': None,
    'vocab_mcq': None,
    'flashcards': None,
    'mindmap': None,
    'cloze': None,
    'match_pairs': None,
}


def _get_fallback_chain(name: str) -> Optional[LLMChain]:
    """Return (and cache) a Gemini-based fallback chain for the given name."""
    if name not in _fallback_chains:
        return None
    if _fallback_chains[name] is not None:
        return _fallback_chains[name]

    try:
        fallback_llm = get_gemini_chat_llm(temperature=0.2)
    except Exception as exc:
        _fallback_chains[name] = None
        return None

    if name == 'summary':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=summary_prompt_template)
    elif name == 'question':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=question_prompt_template)
    elif name == 'mcq':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=mcq_prompt_template)
    elif name == 'vocab_summary_table':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=vocab_summary_table_template)
    elif name == 'vocab_story':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=vocab_story_template)
    elif name == 'vocab_mcq':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=vocab_mcq_template)
    elif name == 'flashcards':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=flashcards_template)
    elif name == 'mindmap':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=mindmap_template)
    elif name == 'cloze':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=cloze_template)
    elif name == 'match_pairs':
        _fallback_chains[name] = LLMChain(llm=fallback_llm, prompt=match_pairs_template)
    return _fallback_chains[name]

JSON_BLOCK_PATTERN = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', re.S | re.M)
MARKDOWN_JSON_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S | re.M)
JSON_ARRAY_PATTERN = re.compile(r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]', re.S | re.M)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def _extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON block từ text response của LLM.
    Hỗ trợ cả markdown code blocks và raw JSON (object hoặc array).
    """
    if not text:
        return None
    
    markdown_match = MARKDOWN_JSON_PATTERN.search(text)
    if markdown_match:
        return markdown_match.group(1)
    
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        return match.group()
    
    array_match = JSON_ARRAY_PATTERN.search(text)
    if array_match:
        return array_match.group()
    
    return None


def _fix_invalid_unicode_escapes(text: str) -> str:
    """
    Fix invalid Unicode escape sequences trong JSON string.
    Thay thế các invalid \\u sequences bằng ký tự an toàn hoặc escape đúng.
    """
    import re
    
    def fix_unicode_escape(match):
        """Fix một \\u escape sequence"""
        seq = match.group(0)
        if len(seq) == 6:
            hex_part = seq[2:6]
            try:
                int(hex_part, 16)
                try:
                    seq.encode('utf-8').decode('unicode_escape')
                    return seq  
                except (UnicodeDecodeError, ValueError):
                    return ' '
            except ValueError:
                return ' '
        return ' '
    
    text = re.sub(r'\\u[0-9a-fA-F]{0,4}', fix_unicode_escape, text)
    
    return text


def _safe_json_loads(payload: str, fallback: Any) -> Any:
    """
    Parse JSON từ LLM response một cách an toàn.
    Hỗ trợ cả markdown code blocks và raw JSON.
    Xử lý invalid escape sequences và các lỗi JSON phổ biến.
    """
    if not payload:
        return fallback
    
    try:
        return json.loads(payload.strip())
    except json.JSONDecodeError:
        pass
    
    json_block = _extract_json_block(payload)
    if json_block:
        try:
                return json.loads(json_block)
        except json.JSONDecodeError:
            pass
        
        try:
            fixed_block = _fix_invalid_unicode_escapes(json_block)
            return json.loads(fixed_block)
        except json.JSONDecodeError:
            pass
        
        try:
            import re
            def safe_unicode_replace(match):
                seq = match.group(0)
                try:
                    return seq.encode('utf-8').decode('unicode_escape')
                except:
                    return ' '
            fixed_block = re.sub(r'\\u[0-9a-fA-F]{4}', safe_unicode_replace, json_block)
            return json.loads(fixed_block)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[summarizer] Extracted block (first 200 chars): {json_block[:200]}")
    
    return fallback


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w-]", "", word or "").lower().strip()


def _is_stopword(word: str) -> bool:
    w = _normalize_word(word)
    return not w or len(w) < 3 or w in STOPWORDS


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = SENTENCE_SPLIT_PATTERN.split(text.replace('\n', ' '))
    return [s.strip() for s in sentences if s.strip()]


def _fallback_summary(raw_text: str) -> Dict[str, Any]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        return {
            'one_sentence': raw_text[:200],
            'short_paragraph': raw_text[:300],
            'bullet_points': [raw_text[:200]]
        }
    
    one_sentence = sentences[0]
    short_paragraph = ' '.join(sentences[:3])
    bullet_points = sentences[:5]
    
    return {
        'one_sentence': one_sentence,
        'short_paragraph': short_paragraph,
        'bullet_points': bullet_points
    }


def _fallback_questions(raw_text: str) -> List[Dict[str, str]]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text[:200]]
    
    questions = []
    for idx, sentence in enumerate(sentences[:10], 1):
        questions.append({
            'question': f"Nội dung quan trọng số {idx} là gì liên quan tới: \"{sentence}\"?",
            'answer': sentence
        })
    
    while len(questions) < 5:
        questions.append({
            'question': f"Hãy nêu lại ý chính số {len(questions)+1} của ghi chú.",
            'answer': sentences[0]
        })
    
    return questions


def _fallback_mcqs(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    sentences = _split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text[:200]]
    
    levels = ['easy', 'medium', 'hard']
    mcqs: Dict[str, List[Dict[str, Any]]] = {}
    option_labels = ['A', 'B', 'C', 'D']
    for idx, level in enumerate(levels):
        fact = sentences[min(idx, len(sentences) - 1)]
        distractors = []
        for offset in range(1, 4):
            source_idx = min(idx + offset, len(sentences) - 1)
            distractor_sentence = sentences[source_idx]
            if distractor_sentence == fact:
                distractor_sentence = f"Cách giải thích khác cho nội dung: {fact}"
            distractors.append(distractor_sentence)
        options = {}
        correct_label = option_labels[idx % len(option_labels)]
        distractor_iter = iter(distractors)
        for label in option_labels:
            if label == correct_label:
                options[label] = fact
            else:
                options[label] = next(distractor_iter)
        mcqs[level] = [{
            'question': f"Dựa trên ghi chú, ý nào mô tả chính xác nhất: \"{fact}\"?",
            'options': options,
            'answer': correct_label,
            'explanation': (
                f"Phương án {correct_label} tái hiện đúng nội dung đã nêu. "
                "Các phương án khác đề cập chi tiết liên quan nhưng không khớp hoàn toàn với ý chính này."
            )
        }]
    return mcqs


async def _run_chain(chain: LLMChain, variables: Dict[str, Any]) -> str:
    """
    Run a LangChain chain và trả về text response.
    Xử lý các exception có thể xảy ra khi invoke chain.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: chain.invoke(variables))
    except Exception as e:
        error_msg = str(e)
        if "Missing some input keys" in error_msg:
            print(f"[summarizer] Chain invoke error (likely malformed LLM response): {error_msg}")
        raise
    
    if isinstance(result, str):
        result_stripped = result.strip()
        if result_stripped.startswith('{') or result_stripped.startswith('['):
            return result
        return result
    if isinstance(result, dict):
        for key in ('text', 'output', 'result'):
            val = result.get(key)
            if isinstance(val, str):
                return val
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)
    return str(result)


async def _run_chain_with_fallback(chain: LLMChain, name: str, variables: Dict[str, Any]) -> str:
    """
    Run a chain; if it fails (quota/timeout/etc.), retry once with fallback LLM.
    Catches "Missing some input keys" errors which can occur when LLM response is malformed.
    """
    try:
        return await _run_chain(chain, variables)
    except Exception as primary_exc:
        error_msg = str(primary_exc)
        if "Missing some input keys" in error_msg:
            print(f"[summarizer] Primary chain '{name}' failed with input keys error, trying fallback: {primary_exc}")
        else:
            print(f"[summarizer] Primary chain '{name}' failed, trying fallback: {primary_exc}")
        
        fallback_chain = _get_fallback_chain(name)
        if not fallback_chain:
            raise
        
        try:
            return await _run_chain(fallback_chain, variables)
        except Exception as fallback_exc:
            print(f"[summarizer] Fallback chain '{name}' also failed: {fallback_exc}")
            raise


def _build_summary_instructions(
    db: Optional[Session],
    raw_text: str,
    file_type: Optional[str],
    use_rag: bool
) -> str:
    instructions = (
        "Bạn là trợ lý AI giúp tạo tài liệu ôn tập tiếng Việt."
        " Hãy sửa lỗi chính tả nếu cần và nêu rõ thông tin trọng tâm."
    )
    
    if file_type:
        instructions += f"\nNgữ cảnh: nội dung được trích từ {file_type.upper()}."
    
    if db and use_rag:
        try:
            from app.services.prompt_retriever import prompt_retriever
            rag_prompt = prompt_retriever.get_contextual_prompt(
                db=db,
                raw_text=raw_text,
                file_type=file_type
            )
            instructions = rag_prompt.strip()
        except Exception as exc:
            print(f"[summarizer] Skip RAG prompt due to error: {exc}")
    
    instructions += (
        "\nĐảm bảo:\n"
        "- one_sentence <= 40 từ\n"
        "- short_paragraph dài 3-5 câu, giữ số liệu quan trọng\n"
        "- bullet_points từ 3-7 ý, mỗi ý ngắn gọn"
    )
    return instructions


async def generate_summary_bundle(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    instructions = _build_summary_instructions(db, raw_text, file_type, use_rag)
    try:
        response = await _run_chain_with_fallback(
            summary_chain,
            'summary',
            {'instructions': instructions, 'raw_text': raw_text}
        )
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict):
            bullets = parsed.get('bullet_points')
            if isinstance(bullets, str):
                parsed['bullet_points'] = [b.strip() for b in bullets.split('\n') if b.strip()]
            return parsed
    except Exception as exc:
        print(f"[summarizer] Error generating summaries: {exc}")
    
    return _fallback_summary(raw_text)


async def generate_question_set(raw_text: str) -> List[Dict[str, str]]:
    try:
        response = await _run_chain_with_fallback(
            question_chain,
            'question',
            {'raw_text': raw_text}
        )
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict) and isinstance(parsed.get('questions'), list):
            return parsed['questions']
    except Exception as exc:
        print(f"[summarizer] Error generating questions: {exc}")
    return _fallback_questions(raw_text)


async def generate_mcq_set(raw_text: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        response = await _run_chain_with_fallback(
            mcq_chain,
            'mcq',
            {'raw_text': raw_text}
        )
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict):
            normalized = {}
            for level in ['easy', 'medium', 'hard']:
                level_questions = parsed.get(level, [])
                if not isinstance(level_questions, list):
                    continue
                normalized[level] = level_questions
            if normalized:
                return normalized
    except Exception as exc:
        print(f"[summarizer] Error generating MCQs: {exc}")
    return _fallback_mcqs(raw_text)


async def generate_learning_assets(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    summaries = await generate_summary_bundle(
        raw_text=raw_text,
        db=db,
        file_type=file_type,
        use_rag=use_rag
    )
    questions = await generate_question_set(raw_text)
    mcqs = await generate_mcq_set(raw_text)
    return {
        'summaries': summaries,
        'questions': questions,
        'mcqs': mcqs
    }

def _parse_vocab_list(raw_text: str, checked_vocab_items: Optional[str]) -> List[str]:
    def filter_phrases(words: List[str]) -> List[str]:
        """
        Keep each checklist item as-is (phrase), up to 15 unique entries.
        Do not split by whitespace; only trim and deduplicate (case-insensitive).
        """
        seen = set()
        filtered: List[str] = []
        for w in words:
            phrase = (w or "").strip()
            if not phrase:
                continue
            if len(phrase.split()) == 1 and _is_stopword(_normalize_word(phrase)):
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(phrase)
            if len(filtered) >= 15:
                break
        return filtered

    def filter_words(words: List[str]) -> List[str]:
        """
        For fallback tokenization from raw_text: normalize, drop stopwords/dupes.
        """
        seen = set()
        filtered: List[str] = []
        for w in words:
            norm = _normalize_word(w)
            if not norm or _is_stopword(norm):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            filtered.append(norm)
            if len(filtered) >= 15:
                break
        return filtered

    vocab_words: List[str] = []
    if checked_vocab_items:
        try:
            parsed = json.loads(checked_vocab_items)
            if isinstance(parsed, list):
                vocab_words = filter_phrases([str(w) for w in parsed])
        except Exception:
            vocab_words = []

        if not vocab_words:
            candidates = []
            separator_found = False
            for separator in ["\n", ";", ","]:
                if separator in checked_vocab_items:
                    for item in checked_vocab_items.split(separator):
                        cand = item.strip()
                if cand:
                    candidates.append(cand)
                    separator_found = True
                    break
            
            if not separator_found:
                cand = checked_vocab_items.strip()
                if cand:
                    candidates.append(cand)
            
            vocab_words = filter_phrases(candidates)
    if not vocab_words:
        tokens = [w.strip(" ,.;:()[]{}\"'") for w in raw_text.split()]
        vocab_words = filter_words(tokens)
    
    if not vocab_words:
        vocab_words = ["vocabulary"]
    
    print(f"[vocab_parse] Parsed vocab_words: {vocab_words}")
    return vocab_words


async def _generate_vocab_summary_table(raw_text: str, vocab_list: List[str]) -> Optional[List[Dict[str, Any]]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    try:
        response = await _run_chain_with_fallback(vocab_summary_table_chain, 'vocab_summary_table', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                word = item.get('word')
                if _is_stopword(word):
                    continue
                translation = item.get('translation') or ""
                if "Nghĩa của" in translation:
                    continue
                collocations = item.get('collocations')
                if isinstance(collocations, list):
                    seen = set()
                    item['collocations'] = [c for c in collocations if not (c in seen or seen.add(c))]
                valid_items.append(item)
            if valid_items:
                return valid_items
    except Exception as exc:
        print(f"[summarizer] Error generating vocab summary table: {exc}")
    return None


async def _generate_vocab_story(raw_text: str, vocab_list: List[str], retry_count: int = 0) -> Optional[Dict[str, Any]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    max_retries = 2
    try:
        response = await _run_chain_with_fallback(vocab_story_chain, 'vocab_story', payload)
        if response and len(response) > 500:
            print(f"[summarizer] Vocab story response (first 500 chars): {response[:500]}")
        else:
            print(f"[summarizer] Vocab story response: {response}")
        
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict) and parsed.get('title') and parsed.get('paragraphs'):
            paragraphs = parsed.get('paragraphs', [])
            print(f"[summarizer] Vocab story received {len(paragraphs) if isinstance(paragraphs, list) else 0} paragraphs")
            if isinstance(paragraphs, list) and len(paragraphs) >= 5:
                valid_paragraphs = []
                for idx, para in enumerate(paragraphs):
                    if isinstance(para, str) and para.strip():
                        sentences = [s.strip() for s in para.split('.') if s.strip()]
                        if len(sentences) >= 2: 
                            valid_paragraphs.append(para)
                        else:
                            print(f"[summarizer] Paragraph {idx+1} rejected: only {len(sentences)} sentences (required: 2+)")
                
                print(f"[summarizer] Vocab story has {len(valid_paragraphs)} valid paragraphs")
                if len(valid_paragraphs) >= 5:
                    parsed['paragraphs'] = valid_paragraphs
                    used_words = parsed.get('used_words') or []
                    cleaned_used = [uw for uw in used_words if isinstance(uw, dict) and not _is_stopword(uw.get('word'))]
                    parsed['used_words'] = cleaned_used
                    print(f"[summarizer] Vocab story accepted with {len(valid_paragraphs)} paragraphs")
                    return parsed
                else:
                    print(f"[summarizer] Vocab story rejected: only {len(valid_paragraphs)} valid paragraphs (required: 5+)")
            else:
                print(f"[summarizer] Vocab story rejected: only {len(paragraphs) if isinstance(paragraphs, list) else 0} paragraphs (required: 5+)")

            if retry_count < max_retries:
                print(f"[summarizer] Retrying vocab story generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_vocab_story(raw_text, vocab_list, retry_count + 1)
        else:
            print(f"[summarizer] Vocab story parse failed or invalid structure")
            if retry_count < max_retries:
                print(f"[summarizer] Retrying vocab story generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_vocab_story(raw_text, vocab_list, retry_count + 1)
    except Exception as exc:
        print(f"[summarizer] Error generating vocab story: {exc}")
        if retry_count < max_retries:
            print(f"[summarizer] Retrying vocab story generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_vocab_story(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_vocab_mcqs(raw_text: str, vocab_list: List[str]) -> Optional[List[Dict[str, Any]]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    try:
        response = await _run_chain_with_fallback(vocab_mcq_chain, 'vocab_mcq', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if not item.get('question') or not item.get('options'):
                    continue
                target = item.get('vocab_target') or ""
                if _is_stopword(target):
                    continue
                valid_items.append(item)
            if valid_items:
                return valid_items
    except Exception as exc:
        print(f"[summarizer] Error generating vocab MCQs: {exc}")
    return None


async def _generate_cloze_tests(raw_text: str, vocab_list: List[str], retry_count: int = 0) -> Optional[List[Dict[str, Any]]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    max_retries = 2
    try:
        response = await _run_chain_with_fallback(cloze_chain, 'cloze', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                paragraph = item.get('paragraph') or ""
                if not paragraph or "___" not in paragraph:
                    continue
                if "This is a ___" in paragraph:
                    continue
                if not item.get('blanks'):
                    continue
                blanks = item.get('blanks')
                if isinstance(blanks, list) and blanks:
                    cleaned_blanks = []
                    for b in blanks:
                        if not isinstance(b, dict):
                            continue
                        ans = b.get('answer')
                        if _is_stopword(ans):
                            continue
                        cleaned_blanks.append(b)
                    if cleaned_blanks and len(cleaned_blanks) == 1:
                        item['blanks'] = cleaned_blanks
                        valid_items.append(item)
                    else:
                        print(f"[summarizer] Cloze test rejected: {len(cleaned_blanks)} blanks in one question (required: 1 blank per question)")
            min_required = min(len(vocab_list), 3) 
            if valid_items and len(valid_items) >= min_required:
                return valid_items
            else:
                print(f"[summarizer] Cloze test rejected: only {len(valid_items)} questions (required: {min_required}+)")
                if retry_count < max_retries:
                    print(f"[summarizer] Retrying cloze test generation (attempt {retry_count + 1}/{max_retries})")
                    return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
        else:
            if retry_count < max_retries:
                print(f"[summarizer] Retrying cloze test generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
    except Exception as exc:
        print(f"[summarizer] Error generating cloze tests: {exc}")
        if retry_count < max_retries:
            print(f"[summarizer] Retrying cloze test generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_match_pairs(raw_text: str, vocab_list: List[str], retry_count: int = 0) -> Optional[List[Dict[str, Any]]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    max_retries = 2
    try:
        response = await _run_chain_with_fallback(match_pairs_chain, 'match_pairs', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                word = item.get('word')
                if _is_stopword(word):
                    continue
                meaning = item.get('meaning') or ""     
                meaning_lower = meaning.lower()
                placeholder_patterns = [
                    "nghĩa của", "nghĩa ngắn gọn", "ý nghĩa ngắn gọn", 
                    "thực tế của", "meaning of", "nghĩa của từ",
                    "nghĩa cụ thể của", "nghĩa tiếng việt của", "nghĩa của x"
                ]
                if any(pattern in meaning_lower for pattern in placeholder_patterns):
                    print(f"[summarizer] Match pairs rejected: placeholder meaning '{meaning}' for word '{word}'")
                    continue
                if not meaning.strip() or len(meaning.strip()) < 2:
                    print(f"[summarizer] Match pairs rejected: empty or too short meaning '{meaning}' for word '{word}'")
                    continue
                if len(meaning.strip()) > 50:  
                    print(f"[summarizer] Match pairs rejected: meaning too long '{meaning}' for word '{word}'")
                    continue
                valid_items.append(item)
            if valid_items:
                if len(valid_items) >= 8:
                    return valid_items[:8]
                elif len(valid_items) >= 4: 
                    return valid_items
                else:
                    print(f"[summarizer] Match pairs rejected: only {len(valid_items)} valid pairs (required: 4+)")
                    if retry_count < max_retries:
                        print(f"[summarizer] Retrying match pairs generation (attempt {retry_count + 1}/{max_retries})")
                        return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
            else:
                if retry_count < max_retries:
                    print(f"[summarizer] Retrying match pairs generation (attempt {retry_count + 1}/{max_retries})")
                    return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
        else:
            if retry_count < max_retries:
                print(f"[summarizer] Retrying match pairs generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
    except Exception as exc:
        print(f"[summarizer] Error generating match pairs: {exc}")
        if retry_count < max_retries:
            print(f"[summarizer] Retrying match pairs generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_flashcards(raw_text: str, vocab_list: List[str]) -> Optional[List[Dict[str, Any]]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    try:
        response = await _run_chain_with_fallback(flashcards_chain, 'flashcards', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_items = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                word = item.get('word') or item.get('front')
                if _is_stopword(word):
                    continue
                valid_items.append(item)
            if valid_items:
                return valid_items
    except Exception as exc:
        print(f"[summarizer] Error generating flashcards: {exc}")
    return None


async def _generate_mindmap(raw_text: str, vocab_list: List[str]) -> Optional[Dict[str, Any]]:
    payload = {
        "raw_text": raw_text,
        "vocab_list": "\n".join(vocab_list),
    }
    try:
        response = await _run_chain_with_fallback(mindmap_chain, 'mindmap', payload)
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict):
            def clean_words(words):
                if not isinstance(words, list):
                    return []
                seen = set()
                out = []
                for w in words:
                    if _is_stopword(w):
                        continue
                    if w in seen:
                        continue
                    seen.add(w)
                    out.append(w)
                return out

            for key in ('by_topic', 'by_difficulty', 'by_pos', 'by_relation'):
                groups = parsed.get(key)
                if isinstance(groups, list):
                    cleaned_groups = []
                    for g in groups:
                        if not isinstance(g, dict):
                            continue
                        name_fields = [g.get('topic'), g.get('group_name'), g.get('description')]
                        if any(str(v).lower() == 'general' or str(v).lower() == 'related_terms' for v in name_fields if v):
                            continue
                        g_words = clean_words(g.get('words'))
                        g['words'] = g_words
                        if g_words:
                            cleaned_groups.append(g)
                    parsed[key] = cleaned_groups
            if any(parsed.get(k) for k in ('by_topic', 'by_difficulty', 'by_pos', 'by_relation')):
                return parsed
    except Exception as exc:
        print(f"[summarizer] Error generating mindmap: {exc}")
    return None


def _fallback_vocab_bundle(vocab_words: List[str]) -> Dict[str, Any]:
    def pick_words(n):
        return vocab_words[:n] if len(vocab_words) >= n else vocab_words

    summary_table = []
    for w in vocab_words[:15]:
        summary_table.append({
            "word": w,
            "translation": f"{w} (nghĩa tiếng Việt cụ thể)",
            "part_of_speech": "noun",
            "definition": f"Short definition of {w} in English (1 sentence).",
            "usage_note": f"Cách dùng {w} trong ngữ cảnh quen thuộc.",
            "common_structures": [f"use {w} to ...", f"{w} + noun"],
            "collocations": [f"{w} example", f"{w} phrase"],
        })

    story_paragraphs = []
    words = pick_words(min(10, len(vocab_words)))
    for i in range(5):
        if i < len(words):
            story_paragraphs.append(
                f"In this story, we explore the concept of **{words[i]}**. "
                f"This word plays an important role in understanding the context. "
                f"Let us see how **{words[i]}** is used in different situations."
            )
        else:
            word = words[i % len(words)] if words else "vocabulary"
            story_paragraphs.append(
                f"Continuing our exploration, we find that **{word}** appears again. "
                f"This repetition helps us remember the word better. "
                f"Understanding **{word}** is key to mastering this vocabulary set."
            )

    story = {
        "title": "Ghi nhớ từ vựng",
        "paragraphs": story_paragraphs,
        "used_words": [{"word": w, "bolded": True} for w in words],
    }

    mcqs = []
    for i, w in enumerate(pick_words(7), start=1):
        mcqs.append({
            "id": i,
            "type": "vocab_mcq",
            "vocab_target": w,
            "question": f"What does '{w}' mean in context?",
            "options": {
                "A": f"The correct meaning of {w}",
                "B": f"A close but wrong meaning of {w}",
                "C": f"An unrelated meaning",
                "D": f"A random option",
            },
            "answer": "A",
            "explanation": f"{w} means the definition in A.",
        })

    flashcards = []
    for w in pick_words(10):
        flashcards.append({
            "word": w,
            "front": w,
            "back": {
                "meaning": f"Meaning of {w} (concise, real).",
                "example": f"A natural example using {w}.",
                "usage_note": f"Usage note for {w}.",
                "synonyms": [f"{w}_syn1"],
                "antonyms": [],
            },
            "srs_schedule": {
                "intervals": [1, 3, 7],
                "recall_task": f"Recall {w} meaning and make a sentence.",
            },
        })

    mindmap = {
        "by_topic": [{"topic": "Context", "description": "Một nhóm theo ngữ cảnh", "words": pick_words(6)}],
        "by_difficulty": [{"level": "easy", "description": "Cơ bản", "words": pick_words(5)}],
        "by_pos": [{"pos": "noun", "words": pick_words(8)}],
        "by_relation": [{"group_name": "related", "description": "Từ liên quan", "words": pick_words(6)}],
    }

    cloze = []
    words_for_cloze = pick_words(min(8, len(vocab_words)))
    for idx, word in enumerate(words_for_cloze, start=1):
        cloze.append({
            "title": f"Vocabulary Practice {idx}",
            "paragraph": f"Fill in the blank: The word '{word}' means ___{idx}___ in this context.",
            "blanks": [
                {"id": idx, "answer": word, "explanation": f"The correct answer is '{word}'.", "on_correct_example": f"Example: I saw a {word} at the beach."}
            ],
        })

    match_pairs = []
    words_for_pairs = pick_words(min(8, len(vocab_words)))
    simple_meanings = {
        "whale": "cá voi",
        "shark": "cá mập", 
        "shrimp": "tôm",
        "squid": "mực",
        "crab": "cua",
        "snail": "ốc sên",
        "dolphin": "cá heo",
        "fish": "cá",
        "ocean": "đại dương",
        "sea": "biển",
        "sand": "cát",
        "rock": "đá"
    }
    for idx, w in enumerate(words_for_pairs, start=1):
        meaning = simple_meanings.get(w.lower(), w)
        match_pairs.append({
            "id": idx,
            "word": w,
            "meaning": meaning,
            "hint": f"Gợi ý ngắn cho {w}"
        })

    return {
        "summary_table": summary_table,
        "vocab_story": story,
        "vocab_mcqs": mcqs,
        "flashcards": flashcards,
        "mindmap": mindmap,
        "cloze_tests": cloze,
        "match_pairs": match_pairs,
    }


async def generate_vocab_bundle(
    raw_text: str,
    checked_vocab_items: Optional[str] = None,
) -> Dict[str, Any]:
    vocab_words = _parse_vocab_list(raw_text, checked_vocab_items)

    summary_table, story, mcqs, flashcards, mindmap, cloze, match_pairs = await asyncio.gather(
        _generate_vocab_summary_table(raw_text, vocab_words),
        _generate_vocab_story(raw_text, vocab_words),
        _generate_vocab_mcqs(raw_text, vocab_words),
        _generate_flashcards(raw_text, vocab_words),
        _generate_mindmap(raw_text, vocab_words),
        _generate_cloze_tests(raw_text, vocab_words),
        _generate_match_pairs(raw_text, vocab_words),
    )

    fallback = _fallback_vocab_bundle(vocab_words)

    if story:
        story_paras = story.get('paragraphs', [])
        if not isinstance(story_paras, list) or len(story_paras) < 5:
            print(f"[summarizer] Final validation: vocab_story rejected ({len(story_paras) if isinstance(story_paras, list) else 0} paragraphs), using fallback")
            story = None
    
    if cloze:
        valid_cloze = []
        for item in cloze:
            if isinstance(item, dict):
                blanks = item.get('blanks', [])
                if isinstance(blanks, list) and len(blanks) == 1:
                    valid_cloze.append(item)
                else:
                    print(f"[summarizer] Final validation: cloze item rejected ({len(blanks) if isinstance(blanks, list) else 0} blanks)")
        if valid_cloze:
            cloze = valid_cloze
        else:
            print(f"[summarizer] Final validation: all cloze_tests rejected, using fallback")
            cloze = None
    
    if match_pairs:
        valid_pairs = []
        for item in match_pairs:
            if isinstance(item, dict):
                meaning = item.get('meaning', '')
                meaning_lower = str(meaning).lower()
                placeholder_patterns = [
                    "nghĩa của", "nghĩa ngắn gọn", "ý nghĩa ngắn gọn", 
                    "thực tế của", "meaning of", "nghĩa của từ"
                ]
                if not any(pattern in meaning_lower for pattern in placeholder_patterns) and meaning.strip():
                    valid_pairs.append(item)
                else:
                    print(f"[summarizer] Final validation: match_pair rejected (placeholder meaning: '{meaning}')")
        if valid_pairs:
            match_pairs = valid_pairs
        else:
            print(f"[summarizer] Final validation: all match_pairs rejected, using fallback")
            match_pairs = None

    return {
        "summary_table": summary_table if summary_table else fallback.get("summary_table"),
        "vocab_story": story if story else fallback.get("vocab_story"),
        "vocab_mcqs": mcqs if mcqs else fallback.get("vocab_mcqs"),
        "flashcards": flashcards if flashcards else fallback.get("flashcards"),
        "mindmap": mindmap if mindmap else fallback.get("mindmap"),
        "cloze_tests": cloze if cloze else fallback.get("cloze_tests"),
        "match_pairs": match_pairs if match_pairs else fallback.get("match_pairs"),
    }


async def summarize_text(
    raw_text: str,
    db: Optional[Session] = None,
    file_type: Optional[str] = None,
    use_rag: bool = True
) -> str:
    """
    Backwards-compatible helper để lấy đoạn tóm tắt chính (3-5 câu).
    """
    summaries = await generate_summary_bundle(
        raw_text=raw_text,
        db=db,
        file_type=file_type,
        use_rag=use_rag
    )
    return summaries.get('short_paragraph') or summaries.get('one_sentence') or ''
