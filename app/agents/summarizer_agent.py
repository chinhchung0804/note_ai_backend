import json
import re
import asyncio
from typing import Optional, Dict, Any, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import Session

from app.agents.llm_config import get_openai_chat_llm

PRIMARY_LLM = get_openai_chat_llm(temperature=0.2)
TRANSLATE_LLM = PRIMARY_LLM  

LLM_TIMEOUT_SECONDS = 300.0

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "up", "about", "into", "over", "after", "under", "above",
    "below", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "but",
}

GLOBAL_VOCAB_RULES = (
    "QUY TẮC TUYỆT ĐỐI (ÁP DỤNG CHO TẤT CẢ OUTPUT):\n"
    "- CHỈ sử dụng đúng các từ có trong vocab_list.\n"
    "- KHÔNG được tự ý thêm, suy đoán, diễn giải lại hoặc thay thế từ vựng bằng từ mới.\n"
    "- Mỗi từ vựng phải được xử lý ĐỘC LẬP, không phụ thuộc các từ khác.\n"
    "- Nếu một từ vựng KHÔNG đáp ứng ĐẦY ĐỦ các yêu cầu, PHẢI BỎ QUA từ đó.\n"
    "- KHÔNG tạo nội dung mâu thuẫn với nghĩa, thời gian, hành động hoặc ngữ cảnh trong raw_text.\n"
    "- KHÔNG dùng placeholder, KHÔNG để trống trường dữ liệu, KHÔNG dùng mô tả mơ hồ.\n"
    "- Tất cả nghĩa tiếng Việt PHẢI chính xác và đầy đủ (KHÔNG để nguyên tiếng Anh).\n"
    "- Output BẮT BUỘC là JSON thuần hợp lệ (dùng double quotes, KHÔNG markdown, KHÔNG text thừa).\n"
)

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
        "- Mỗi độ khó (easy, medium, hard) tạo từ 3-5 câu hỏi\n"
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
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "Bạn là chuyên gia dạy từ vựng tiếng Anh cho người học Việt Nam. "
        "Tạo bảng tóm tắt từ vựng CHI TIẾT – THỰC TẾ – DÙNG ĐƯỢC.\n\n"

        "FORMAT (BẮT BUỘC):\n"
        "- Trả về CHỈ JSON thuần (mảng), KHÔNG markdown, KHÔNG text thừa\n"
        "- Unicode hợp lệ, dùng double quotes\n\n"

        "QUAN TRỌNG - SỬ DỤNG CONTEXT:\n"
        "- raw_text chứa ngữ cảnh thực tế của bài học\n"
        "- Nếu từ trong vocab_list CÓ trong raw_text: sử dụng ngữ cảnh đó để giải thích\n"
        "- Nếu từ trong vocab_list KHÔNG có trong raw_text: giải thích nghĩa từ điển CHÍNH XÁC\n"
        "- Translation PHẢI là nghĩa tiếng Việt CHÍNH XÁC (ví dụ: hand = bàn tay, foot = bàn chân, không được để nguyên 'hand')\n\n"

        "YÊU CẦU NỘI DUNG:\n"
        "- Số lượng mục PHẢI bằng số lượng từ trong vocab_list (trừ khi vocab_list trống)\n"
        "- KHÔNG được suy luận hoặc tạo thêm từ\n"
        "- Ưu tiên vocab_list; nếu trống, trích từ khóa chính trong raw_text\n"
        "- Loại bỏ stopwords (the, and, with, of, ...)\n"
        "- KHÔNG dùng placeholder\n"
        "- Translation: nghĩa tiếng Việt CHÍNH XÁC (bắt buộc phải dịch, không được để nguyên tiếng Anh)\n"
        "- Phonetic: phiên âm IPA nếu có (ví dụ: /hænd/ cho hand). Nếu không rõ, bỏ trống.\n"
        "- Definition: định nghĩa tiếng Việt 1–2 câu, đầy đủ\n"
        "- Usage_note: cách dùng CỤ THỂ trong đời thực\n"
        "- Common_structures & Collocations: dạng NGƯỜI BẢN NGỮ hay dùng\n\n"

        "Schema JSON (giữ nguyên dấu ngoặc):\n"
        "{% raw %}[\n"
        "  {\n"
        "    \"word\": \"từ\",\n"
        "    \"translation\": \"nghĩa tiếng Việt\",\n"
        "    \"phonetic\": \"/phiên âm IPA/\",\n"
        "    \"part_of_speech\": \"noun/verb/adj/adv\",\n"
        "    \"definition\": \"định nghĩa tiếng Việt\",\n"
        "    \"usage_note\": \"hướng dẫn dùng thực tế\",\n"
        "    \"common_structures\": [\"cấu trúc thực tế\"],\n"
        "    \"collocations\": [\"cụm từ thực tế\"]\n"
        "  }\n"
        "]\n{% endraw %}\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text (ngữ cảnh thực tế):\n{{ raw_text }}"
    )
)

vocab_story_template = PromptTemplate(
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "Bạn là người viết truyện giúp người học ghi nhớ từ vựng tiếng Anh.\n"
        "Hãy viết một CÂU CHUYỆN NGẮN, RÕ RÀNG **BẰNG TIẾNG ANH 100%** sử dụng tự nhiên các từ trong vocab_list.\n\n"

        "YÊU CẦU:\n"
        "- Nội dung câu chuyện PHẢI là tiếng Anh tự nhiên (KHÔNG chèn tiếng Việt; các từ vựng bản thân đã là tiếng Anh).\n"
        "- KHÔNG được đưa vào các đối tượng, hành động hoặc khái niệm trái ngược hoàn toàn với nghĩa của từ vựng.\n"
        "- BẮT BUỘC: Phải có ÍT NHẤT 4 đoạn văn (paragraphs), mỗi đoạn 2–4 câu đầy đủ. KHÔNG được ít hơn 4 đoạn.\n"
        "- Câu chuyện nên liên quan và không mâu thuẫn với ngữ cảnh trong raw_text (nếu có).\n"
        "- TẤT CẢ các từ trong vocab_list phải xuất hiện và được **bôi đậm** đúng chỗ sử dụng; mỗi từ tối đa 2 lần.\n"
        "- CHỈ trả về JSON thuần, không có markdown bên ngoài JSON.\n\n"

        "Schema JSON (GIỮ NGUYÊN):\n"
        "{% raw %}{\n"
        '  "title": "Tiêu đề tiếng Anh ngắn gọn",\n'
        '  "paragraphs": ["Đoạn tiếng Anh có **từ**", "..."],\n'
        '  "used_words": [{"word": "từ", "bolded": true}]\n'
        "}\n{% endraw %}\n\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text (ngữ cảnh nếu có):\n{{ raw_text }}"
    )
)

vocab_mcq_template = PromptTemplate(
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "You are an English test item writer creating HIGH-QUALITY vocabulary MCQs.\n\n"

        "MANDATORY RULES:\n"
        "- Each vocab item → EXACTLY 2 questions:\n"
        "  (1) question_type = \"meaning\" → definition-based question\n"
        "  (2) question_type = \"context\" → sentence-based usage question\n"
        "- If BOTH questions cannot be made CLEARLY DIFFERENT → SKIP that vocab.\n\n"

        "QUESTION DESIGN RULES:\n"
        "MEANING QUESTION:\n"
        "- Ask about definition, synonym, or concept.\n"
        "- MUST follow patterns used in real exams, e.g.:\n"
        "  • \"Which word best describes …?\"\n"
        "  • \"Which word means …?\"\n"
        "- DO NOT mention any sentence or situation.\n\n"

        "CONTEXT QUESTION:\n"
        "- MUST include a short English sentence with a blank.\n"
        "- Ask which word best completes the sentence.\n"
        "- Sentence MUST be realistic and similar to school / TOEIC-style questions.\n\n"

        "OPTIONS RULES:\n"
        "- 4 options A/B/C/D.\n"
        "- ALL options must be English words from vocab_list.\n"
        "- ONLY ONE correct answer.\n"
        "- Distractors must be plausible but incorrect.\n\n"

        "EXPLANATION RULES:\n"
        "- Explanation MUST teach:\n"
        "  • For meaning: explain core meaning + why others are wrong.\n"
        "  • For context: explain why it fits the sentence context.\n"
        "- Written in Vietnamese, 1–2 clear sentences.\n\n"

        "OUTPUT FORMAT:\n"
        "- JSON ONLY, no markdown, no extra text.\n\n"

        "Schema JSON:\n"
        "{% raw %}[\n"
        "  {\n"
        "    \"id\": 1,\n"
        "    \"type\": \"vocab_mcq\",\n"
        "    \"question_type\": \"meaning\",\n"
        "    \"vocab_target\": \"word\",\n"
        "    \"question\": \"English exam-style question\",\n"
        "    \"options\": {\"A\": \"\", \"B\": \"\", \"C\": \"\", \"D\": \"\"},\n"
        "    \"answer\": \"A | B | C | D\",\n"
        "    \"explanation\": \"Giải thích nghĩa và phân biệt phương án sai\",\n"
        "    \"when_wrong\": \"Lỗi hiểu nghĩa thường gặp\"\n"
        "  },\n"
        "  {\n"
        "    \"id\": 2,\n"
        "    \"type\": \"vocab_mcq\",\n"
        "    \"question_type\": \"context\",\n"
        "    \"vocab_target\": \"word\",\n"
        "    \"question\": \"Choose the word that best completes the sentence\",\n"
        "    \"options\": {\"A\": \"\", \"B\": \"\", \"C\": \"\", \"D\": \"\"},\n"
        "    \"answer\": \"A | B | C | D\",\n"
        "    \"explanation\": \"Giải thích vì sao từ này phù hợp ngữ cảnh\",\n"
        "    \"when_wrong\": \"Lỗi chọn sai do hiểu sai ngữ cảnh\"\n"
        "  }\n"
        "]\n{% endraw %}\n\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text:\n{{ raw_text }}"
    )
)

flashcards_template = PromptTemplate(
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "Bạn là AI tạo flashcards SRS học từ vựng.\n\n"

        "QUAN TRỌNG - SỬ DỤNG CONTEXT:\n"
        "- raw_text chứa ngữ cảnh thực tế của bài học\n"
        "- Meaning PHẢI là nghĩa tiếng Việt CHÍNH XÁC (bắt buộc phải dịch, không được để nguyên tiếng Anh)\n"
        "- Example có thể liên quan đến raw_text nếu phù hợp\n\n"

        "FORMAT:\n"
        "- Trả về CHỈ JSON (mảng)\n"
        "- KHÔNG markdown, KHÔNG text thừa\n\n"

        "YÊU CẦU:\n"
        "- Chỉ bao gồm các từ đồng nghĩa nếu chúng là những từ thông dụng ở trình độ A2–B1.\n"
        "- Meaning tiếng Việt CHÍNH XÁC (bắt buộc phải dịch, ví dụ: hand = bàn tay, foot = bàn chân)\n"
        "- Example thực tế, đúng ngữ pháp\n"
        "- Usage_note cụ thể\n"
        "- Synonyms / Antonyms thực tế (nếu có)\n"
        "- Có recall_task\n\n"

        "Schema JSON:\n"
        "{% raw %}[\n"
        "  {\n"
        "    \"word\": \"từ\",\n"
        "    \"front\": \"từ\",\n"
        "    \"back\": {\n"
        "      \"meaning\": \"nghĩa tiếng Việt\",\n"
        "      \"example\": \"ví dụ\",\n"
        "      \"usage_note\": \"cách dùng\",\n"
        "      \"synonyms\": [],\n"
        "      \"antonyms\": []\n"
        "    },\n"
        "    \"srs_schedule\": {\n"
        "      \"intervals\": [1,3,7,14],\n"
        "      \"recall_task\": \"nhiệm vụ gợi nhớ\"\n"
        "    }\n"
        "  }\n"
        "]\n{% endraw %}\n\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text (ngữ cảnh thực tế):\n{{ raw_text }}"
    )
)

cloze_template = PromptTemplate(
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "You are an English test writer creating NATURAL cloze test sentences.\n\n"

        "MANDATORY RULES:\n"
        "- Each vocab item → EXACTLY 2 DIFFERENT cloze sentences:\n"
        "  (1) type = \"basic_usage\"\n"
        "  (2) type = \"context_usage\"\n"
        "- If both sentences are not clearly different in purpose → SKIP vocab.\n\n"

        "BASIC USAGE SENTENCE:\n"
        "- General, everyday English.\n"
        "- No specific time, place, or storyline.\n"
        "- Similar to textbook or grammar exercises.\n\n"

        "CONTEXT USAGE SENTENCE:\n"
        "- Must include situation, time, reason, or action.\n"
        "- Clearly linked to raw_text context.\n"
        "- Similar to exam reading-based cloze questions.\n\n"

        "SENTENCE RULES:\n"
        "- English only.\n"
        "- Exactly ONE blank ___1___.\n"
        "- Sentence must be 100% correct when filled.\n\n"

        "EXPLANATION RULES:\n"
        "- Vietnamese explanation MUST explain WHY this word fits.\n"
        "- Mention meaning + usage condition.\n\n"

        "OUTPUT FORMAT:\n"
        "- JSON ONLY, no markdown.\n\n"

        "Schema JSON:\n"
        "{% raw %}[\n"
        "  {\n"
        "    \"vocab\": \"word\",\n"
        "    \"type\": \"basic_usage\",\n"
        "    \"paragraph\": \"English sentence with ___1___\",\n"
        "    \"blanks\": [{\n"
        "      \"id\": 1,\n"
        "      \"answer\": \"word\",\n"
        "      \"explanation\": \"Giải thích nghĩa và cách dùng cơ bản\",\n"
        "      \"on_correct_example\": \"Ví dụ tiếng Việt minh họa\"\n"
        "    }]\n"
        "  },\n"
        "  {\n"
        "    \"vocab\": \"word\",\n"
        "    \"type\": \"context_usage\",\n"
        "    \"paragraph\": \"English context sentence with ___1___\",\n"
        "    \"blanks\": [{\n"
        "      \"id\": 1,\n"
        "      \"answer\": \"word\",\n"
        "      \"explanation\": \"Giải thích dựa trên ngữ cảnh cụ thể\",\n"
        "      \"on_correct_example\": \"Ví dụ khác cùng tình huống\"\n"
        "    }]\n"
        "  }\n"
        "]\n{% endraw %}\n\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text:\n{{ raw_text }}"
    )
)

match_pairs_template = PromptTemplate(
    input_variables=["raw_text", "vocab_list"],
    template_format="jinja2",
    template=(
        GLOBAL_VOCAB_RULES +
        "Tạo trò chơi nối từ – nghĩa (Các cặp từ vựng - nghĩa tiếng Việt).\n\n"

        "YÊU CẦU NGẮN GỌN:\n"
        "- Mỗi từ PHẢI xuất hiện chính xác một lần\n"
        "- Nghĩa KHÔNG ĐƯỢC lặp lại từ tiếng Anh hoặc là cách diễn đạt lại từ đó.\n"
        "- Nghĩa tiếng Việt ngắn (1–3 từ), chính xác, không placeholder.\n"
        "- Chỉ dùng từ trong vocab_list.\n"
        "- Chỉ trả về JSON, không markdown.\n\n"

        "Schema JSON:\n"
        "{% raw %}[\n"
        "  {\"id\": 1, \"word\": \"từ\", \"meaning\": \"nghĩa tiếng Việt\", \"hint\": \"gợi ý\"}\n"
        "]\n{% endraw %}\n\n"

        "vocab_list:\n{{ vocab_list }}\n\n"
        "raw_text (ngữ cảnh thực tế):\n{{ raw_text }}"
    )
)

vocab_summary_table_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_summary_table_template)
vocab_story_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_story_template)
vocab_mcq_chain = LLMChain(llm=PRIMARY_LLM, prompt=vocab_mcq_template)
flashcards_chain = LLMChain(llm=PRIMARY_LLM, prompt=flashcards_template)
cloze_chain = LLMChain(llm=PRIMARY_LLM, prompt=cloze_template)
match_pairs_chain = LLMChain(llm=PRIMARY_LLM, prompt=match_pairs_template)

JSON_BLOCK_PATTERN = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', re.S | re.M)
MARKDOWN_JSON_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S | re.M)
JSON_ARRAY_PATTERN = re.compile(r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]', re.S | re.M)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def _extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON block từ text response của LLM.
    Hỗ trợ cả markdown code blocks và raw JSON (object hoặc array).
    Loại bỏ text giải thích trước JSON.
    """
    if not text:
        return None
    
    markdown_match = MARKDOWN_JSON_PATTERN.search(text)
    if markdown_match:
        return markdown_match.group(1)
    
    first_brace = text.find('{')
    if first_brace >= 0:
        brace_count = 0
        last_brace = -1
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_brace = i
                    break
        
        if last_brace > first_brace:
            json_candidate = text[first_brace:last_brace + 1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
    
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

async def translate_text_via_llm(text: str, target_lang: str = "vi") -> str:
    """
    Dịch nhanh qua OPENAI_MODEL (gpt-4o-mini). Chỉ trả về nội dung dịch, không giải thích.
    """
    if not text:
        return ""
    prompt = (
        "Translate the following text into {target_lang}.\n"
        "- Keep formatting (newlines, **bold**, lists) intact.\n"
        "- Do NOT add explanations or pre/suffix.\n\n"
        "TEXT:\n{text}"
    ).format(target_lang=target_lang, text=text)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: TRANSLATE_LLM.invoke(prompt))
        if hasattr(result, "content"):
            return result.content.strip()
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for k in ("content", "text", "output", "result"):
                val = result.get(k)
                if isinstance(val, str):
                    return val.strip()
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as exc:
        print(f"[translate] error: {exc}")
        return ""


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
    
    Gọi LLM trực tiếp thay vì qua chain để tránh LangChain parse JSON response như template.
    """
    validated_vars = {}
    for key, value in variables.items():
        if value is None:
            validated_vars[key] = ""
        elif isinstance(value, str):
            validated_vars[key] = value.strip()
        else:
            validated_vars[key] = value
    
    var_summary = {k: f"str({len(str(v))})" if isinstance(v, str) else type(v).__name__ for k, v in validated_vars.items()}
    print(f"[summarizer] _run_chain: Input variables: {var_summary}")
    
    loop = asyncio.get_running_loop()
    
    try:
        formatted_prompt = chain.prompt.format(**validated_vars)
        
        llm = chain.llm
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: llm.invoke(formatted_prompt)),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        
        if hasattr(result, 'content'):
            return result.content.strip()
        elif isinstance(result, str):
            return result.strip()
        elif isinstance(result, dict):
            for key in ('content', 'text', 'output', 'result'):
                val = result.get(key)
                if isinstance(val, str):
                    return val.strip()
            try:
                return json.dumps(result, ensure_ascii=False)
            except Exception:
                return str(result)
        else:
            return str(result).strip()
            
    except asyncio.TimeoutError:
        print(f"[summarizer] LLM invoke timed out after {LLM_TIMEOUT_SECONDS}s")
        raise Exception(f"LLM request timed out after {int(LLM_TIMEOUT_SECONDS)} seconds")
    except Exception as e:
        error_msg = str(e)
        print(f"[summarizer] Direct LLM call failed, trying chain.invoke() as fallback: {error_msg[:200]}")
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: chain.invoke(validated_vars)),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            if isinstance(result, dict):
                for key in ('text', 'output', 'result', 'content'):
                    val = result.get(key)
                    if isinstance(val, str):
                        return val.strip()
                try:
                    return json.dumps(result, ensure_ascii=False)
                except Exception:
                    return str(result)
            elif hasattr(result, 'content'):
                return result.content.strip()
            return str(result).strip()
        except Exception as e2:
            print(f"[summarizer] Chain.invoke() fallback also failed: {str(e2)[:200]}")
            raise e  

async def _run_chain_with_fallback(chain: LLMChain, name: str, variables: Dict[str, Any]) -> str:
    """
    Run a chain with OpenAI/MegaLLM (openai-gpt-oss-20b) only.
    
    NOTE: Gemini fallback is DISABLED to avoid quota issues and excessive logging.
    If OpenAI/MegaLLM fails, the exception will be raised directly.
    
    Catches "Missing some input keys" errors which can occur when LLM response is malformed.
    """
    try:
        return await _run_chain(chain, variables)
    except Exception as primary_exc:
        error_msg = str(primary_exc)
        
        is_rate_limit = (
            "429" in error_msg or 
            "rate_limit" in error_msg.lower() or 
            "ResourceExhausted" in error_msg or
            "quota" in error_msg.lower()
        )
        
        if "Missing some input keys" in error_msg:
            print(f"[summarizer] Chain '{name}' failed with input keys error: {error_msg[:200]}")
        elif is_rate_limit:
            print(f"[summarizer] Chain '{name}' failed with rate limit (429)")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"[summarizer] Chain '{name}' timed out after {LLM_TIMEOUT_SECONDS}s")
        else:
            print(f"[summarizer] Chain '{name}' failed: {error_msg[:200]}")
        
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

def normalize_vocab_list(vocab_words: List[str]) -> List[str]:
    """Chuẩn hóa vocab: strip, dedup, sửa lỗi OCR phổ biến."""
    corrections = {
        "hammed shark": "hammerhead shark",
        "hammer shark": "hammerhead shark",
        "yatch": "yacht",
    }
    seen = set()
    cleaned: List[str] = []
    for w in vocab_words:
        if not isinstance(w, str):
            continue
        cand = w.strip()
        if not cand:
            continue
        low = cand.lower()
        if low in corrections:
            cand = corrections[low]
            low = cand.lower()
        if low in seen:
            continue
        if len(cand.split()) == 1 and _is_stopword(cand):
            continue
        seen.add(low)
        cleaned.append(cand)
    return cleaned[:25]

def _parse_vocab_list(raw_text: str, checked_vocab_items: Optional[str]) -> List[str]:
    def filter_phrases(words: List[str]) -> List[str]:
        """
        Keep each checklist item as-is (phrase), up to 25 unique entries.
        Do not split by whitespace; only trim and deduplicate (case-insensitive).
        """
        seen = set()
        filtered: List[str] = []
        for w in words:
            phrase = (w or "").strip()
            if not phrase:
                continue
            if len(phrase.split()) == 1:
                normalized_phrase = _normalize_word(phrase)
                if normalized_phrase in STOPWORDS or (len(normalized_phrase) < 3 and normalized_phrase in STOPWORDS):
                    continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(phrase)
            if len(filtered) >= 25:
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
            if len(filtered) >= 25:
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
    vocab_list_str = "\n".join(vocab_list) if vocab_list else ""
    payload = {
        "raw_text": raw_text or "",
        "vocab_list": vocab_list_str,
    }
    print(f"[summarizer] _generate_vocab_summary_table: raw_text length={len(raw_text)}, vocab_list count={len(vocab_list)}, vocab_list_str length={len(vocab_list_str)}")
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
    if not vocab_list:
        print(f"[summarizer] Vocab story: vocab_list is empty, cannot generate story")
        return None
    
    print(f"[summarizer] Vocab story: vocab_list has {len(vocab_list)} words: {vocab_list[:4]}...")
    
    vocab_list_str = "\n".join(vocab_list) if vocab_list else ""
    payload = {
        "raw_text": raw_text or "",
        "vocab_list": vocab_list_str,
    }
    print(f"[summarizer] _generate_vocab_story: raw_text length={len(raw_text)}, vocab_list_str length={len(vocab_list_str)}")
    max_retries = 2
    try:
        response = await _run_chain_with_fallback(vocab_story_chain, 'vocab_story', payload)
        # Log response để debug
        if response and len(response) > 500:
            print(f"[summarizer] Vocab story response (first 500 chars): {response[:500]}")
        else:
            print(f"[summarizer] Vocab story response: {response}")
        
        parsed = _safe_json_loads(response, None)
        if isinstance(parsed, dict) and parsed.get('title') and parsed.get('paragraphs'):
            paragraphs = parsed.get('paragraphs', [])
            print(f"[summarizer] Vocab story received {len(paragraphs) if isinstance(paragraphs, list) else 0} paragraphs")
            # Yêu cầu tối thiểu: Phải có ít nhất 4 đoạn hợp lệ (theo prompt yêu cầu)
            if isinstance(paragraphs, list) and len(paragraphs) >= 4:
                # Kiểm tra độ dài mỗi đoạn (tối thiểu 1 câu, khuyến nghị 2 câu)
                valid_paragraphs = []
                for idx, para in enumerate(paragraphs):
                    if isinstance(para, str) and para.strip():
                        # Đếm số câu (dựa vào dấu chấm, dấu chấm hỏi, dấu chấm than)
                        import re
                        sentence_endings = re.split(r'[.!?]+', para)
                        sentences = [s.strip() for s in sentence_endings if s.strip() and len(s.strip()) > 4]
                        if len(sentences) >= 1:  # Tối thiểu 1 câu mỗi đoạn
                            valid_paragraphs.append(para)
                        else:
                            print(f"[summarizer] Paragraph {idx+1} rejected: only {len(sentences)} sentences (required: 1+)")
                
                print(f"[summarizer] Vocab story has {len(valid_paragraphs)} valid paragraphs")
                # Chấp nhận story nếu có ít nhất 4 đoạn hợp lệ (theo yêu cầu prompt)
                if len(valid_paragraphs) >= 4:
                    parsed['paragraphs'] = valid_paragraphs
                    used_words = parsed.get('used_words') or []
                    cleaned_used = [uw for uw in used_words if isinstance(uw, dict) and not _is_stopword(uw.get('word'))]
                    parsed['used_words'] = cleaned_used
                    print(f"[summarizer] Vocab story accepted with {len(valid_paragraphs)} paragraphs")
                    return parsed
                else:
                    print(f"[summarizer] Vocab story rejected: no valid paragraphs (all paragraphs were too short)")
            else:
                # Nếu paragraphs không phải list hoặc rỗng, reject
                print(f"[summarizer] Vocab story rejected: invalid paragraphs format (expected list, got {type(paragraphs).__name__}) or empty")
            
            # Retry nếu chưa đạt yêu cầu
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
        error_msg = str(exc)
        is_rate_limit = (
            "429" in error_msg or 
            "rate_limit" in error_msg.lower() or 
            "ResourceExhausted" in error_msg or
            "quota" in error_msg.lower()
        )
        # Không retry nếu là rate limit (sẽ dùng fallback thay vì retry)
        if retry_count < max_retries and not is_rate_limit:
            print(f"[summarizer] Retrying vocab story generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_vocab_story(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_vocab_mcqs(raw_text: str, vocab_list: List[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Generate vocab MCQs in chunks to avoid long single-call latency/timeouts.
    For each vocab word: require 2 questions (meaning + context).
    Also force numeric `id` to avoid client-side NumberFormatException (Android/Java).
    """
    if not vocab_list:
        return None

    def _norm_key(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    # Keep raw_text bounded to reduce token usage and avoid timeouts in large combined notes
    raw_text_for_prompt = (raw_text or "")[:2000]

    chunk_size = 6  # target 5–7; using 6 as default
    chunks = _chunk_list(vocab_list, chunk_size)
    print(
        f"[summarizer] _generate_vocab_mcqs: raw_text length={len(raw_text_for_prompt)}, "
        f"vocab_list count={len(vocab_list)}, chunks={len(chunks)}, chunk_size={chunk_size}"
    )

    sem = asyncio.Semaphore(2)  # limit concurrency to avoid overwhelming the LLM

    async def _run_chunk(chunk_words: List[str], chunk_idx: int) -> List[Dict[str, Any]]:
        async with sem:
            vocab_list_str = "\n".join(chunk_words)
            payload = {
                "raw_text": raw_text_for_prompt,
                "vocab_list": vocab_list_str,
            }
            print(
                f"[summarizer] vocab_mcq chunk {chunk_idx + 1}/{len(chunks)}: "
                f"vocab_count={len(chunk_words)}, vocab_list_str length={len(vocab_list_str)}"
            )
            try:
                response = await _run_chain_with_fallback(vocab_mcq_chain, "vocab_mcq", payload)
                parsed = _safe_json_loads(response, None)
                if not isinstance(parsed, list) or not parsed:
                    return []

                # Basic validation
                candidates: List[Dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    if not item.get("question") or not item.get("options"):
                        continue
                    qtype = (item.get("question_type") or "").strip().lower()
                    if qtype not in ("meaning", "context"):
                        continue
                    target = item.get("vocab_target") or ""
                    if _is_stopword(target):
                        continue
                    candidates.append(item)

                if not candidates:
                    return []

                # Enforce 2 questions per vocab_target (meaning + context)
                grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for item in candidates:
                    target_key = _norm_key(item.get("vocab_target") or "")
                    qtype = (item.get("question_type") or "").strip().lower()
                    grouped.setdefault(target_key, {})[qtype] = item

                out: List[Dict[str, Any]] = []
                for w in chunk_words:
                    key = _norm_key(w)
                    pair = grouped.get(key) or {}
                    if "meaning" in pair and "context" in pair:
                        # Keep stable order: meaning then context
                        out.append(pair["meaning"])
                        out.append(pair["context"])
                return out
            except Exception as exc:
                print(f"[summarizer] Error generating vocab MCQs chunk {chunk_idx + 1}: {exc}")
                return []

    # Run chunks (bounded concurrency) then merge
    chunk_results = await asyncio.gather(*[_run_chunk(c, i) for i, c in enumerate(chunks)])
    merged: List[Dict[str, Any]] = [item for sub in chunk_results for item in sub if isinstance(item, dict)]

    if not merged:
        return None

    # Final normalization: numeric id + stable uid
    for idx, item in enumerate(merged, start=1):
        qtype = (item.get("question_type") or "meaning").strip().lower()
        target = (item.get("vocab_target") or "").strip()
        original_id = item.get("id")
        # Preserve original id as uid if it's useful, otherwise create one from target+type
        if original_id is not None:
            item["uid"] = str(original_id)
        else:
            item["uid"] = f"{target}_{qtype}".strip()
        item["id"] = idx

    print(f"[summarizer] Vocab MCQs merged: {len(merged)} questions (expected up to {len(vocab_list) * 2})")
    return merged


async def _generate_cloze_tests(raw_text: str, vocab_list: List[str], retry_count: int = 0) -> Optional[List[Dict[str, Any]]]:
    # Validate và prepare input variables
    vocab_list_str = "\n".join(vocab_list) if vocab_list else ""
    payload = {
        "raw_text": raw_text or "",
        "vocab_list": vocab_list_str,
    }
    print(f"[summarizer] _generate_cloze_tests: raw_text length={len(raw_text)}, vocab_list_str length={len(vocab_list_str)}")
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
                # Đếm số lượng blanks trong paragraph text (___1___, ___2___, etc.)
                import re
                blank_count_in_text = len(re.findall(r'___\d+___', paragraph))
                if blank_count_in_text != 1:
                    print(f"[summarizer] Cloze test rejected: paragraph has {blank_count_in_text} blanks in text (required: exactly 1 blank per paragraph)")
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
                    # BẮT BUỘC: Mỗi câu hỏi chỉ có 1 blank (format mới)
                    if cleaned_blanks and len(cleaned_blanks) == 1:
                        item['blanks'] = cleaned_blanks
                        valid_items.append(item)
                    else:
                        # Reject format cũ (nhiều blanks trong một paragraph)
                        print(f"[summarizer] Cloze test rejected: {len(cleaned_blanks)} blanks in blanks array (required: 1 blank per question)")
            # Yêu cầu: Tất cả các từ trong vocab_list phải có câu hỏi riêng (mỗi từ một câu hỏi)
            min_required = min(len(vocab_list), 3)  # Tối thiểu 3 câu hỏi
            if valid_items and len(valid_items) >= min_required:
                return valid_items
            else:
                print(f"[summarizer] Cloze test rejected: only {len(valid_items)} questions (required: {min_required}+)")
                # Retry nếu chưa đạt yêu cầu
                if retry_count < max_retries:
                    print(f"[summarizer] Retrying cloze test generation (attempt {retry_count + 1}/{max_retries})")
                    return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
        else:
            if retry_count < max_retries:
                print(f"[summarizer] Retrying cloze test generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
    except Exception as exc:
        print(f"[summarizer] Error generating cloze tests: {exc}")
        error_msg = str(exc)
        is_rate_limit = (
            "429" in error_msg or 
            "rate_limit" in error_msg.lower() or 
            "ResourceExhausted" in error_msg or
            "quota" in error_msg.lower()
        )
        # Không retry nếu là rate limit (sẽ dùng fallback thay vì retry)
        if retry_count < max_retries and not is_rate_limit:
            print(f"[summarizer] Retrying cloze test generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_cloze_tests(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_match_pairs(raw_text: str, vocab_list: List[str], retry_count: int = 0) -> Optional[List[Dict[str, Any]]]:
    # Validate và prepare input variables
    vocab_list_str = "\n".join(vocab_list) if vocab_list else ""
    payload = {
        "raw_text": raw_text or "",
        "vocab_list": vocab_list_str,
    }
    print(f"[summarizer] _generate_match_pairs: raw_text length={len(raw_text)}, vocab_list_str length={len(vocab_list_str)}")
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
                # Reject các placeholder và nghĩa không cụ thể
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
                # Nghĩa phải là từ/cụm từ cụ thể, không phải câu dài
                if len(meaning.strip()) > 50:  # Nghĩa quá dài có thể là placeholder
                    print(f"[summarizer] Match pairs rejected: meaning too long '{meaning}' for word '{word}'")
                    continue
                valid_items.append(item)
            if valid_items:
                return valid_items  # giữ tất cả cặp hợp lệ để hiển thị/ luyện nhiều vòng
            else:
                print(f"[summarizer] Match pairs rejected: 0 valid pairs")
                if retry_count < max_retries:
                    print(f"[summarizer] Retrying match pairs generation (attempt {retry_count + 1}/{max_retries})")
                    return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
        else:
            if retry_count < max_retries:
                print(f"[summarizer] Retrying match pairs generation (attempt {retry_count + 1}/{max_retries})")
                return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
    except Exception as exc:
        print(f"[summarizer] Error generating match pairs: {exc}")
        error_msg = str(exc)
        is_rate_limit = (
            "429" in error_msg or 
            "rate_limit" in error_msg.lower() or 
            "ResourceExhausted" in error_msg or
            "quota" in error_msg.lower()
        )
        # Không retry nếu là rate limit (sẽ dùng fallback thay vì retry)
        if retry_count < max_retries and not is_rate_limit:
            print(f"[summarizer] Retrying match pairs generation after error (attempt {retry_count + 1}/{max_retries})")
            return await _generate_match_pairs(raw_text, vocab_list, retry_count + 1)
    return None


async def _generate_flashcards(raw_text: str, vocab_list: List[str]) -> Optional[List[Dict[str, Any]]]:
    # Validate và prepare input variables
    vocab_list_str = "\n".join(vocab_list) if vocab_list else ""
    payload = {
        "raw_text": raw_text or "",
        "vocab_list": vocab_list_str,
    }
    print(f"[summarizer] _generate_flashcards: raw_text length={len(raw_text)}, vocab_list_str length={len(vocab_list_str)}")
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
    return None


def _fallback_vocab_bundle(vocab_words: List[str]) -> Dict[str, Any]:
    return {
        "summary_table": [],
        "vocab_story": None,
        "vocab_mcqs": [],
        "flashcards": [],
        "mindmap": None,
        "cloze_tests": [],
        "match_pairs": [],
    }


async def generate_vocab_bundle(
    raw_text: str,
    checked_vocab_items: Optional[str] = None,
) -> Dict[str, Any]:
    vocab_words = normalize_vocab_list(_parse_vocab_list(raw_text, checked_vocab_items))

    summary_table, story, mcqs, flashcards, cloze, match_pairs = await asyncio.gather(
        _generate_vocab_summary_table(raw_text, vocab_words),
        _generate_vocab_story(raw_text, vocab_words),
        _generate_vocab_mcqs(raw_text, vocab_words),
        _generate_flashcards(raw_text, vocab_words),
        _generate_cloze_tests(raw_text, vocab_words),
        _generate_match_pairs(raw_text, vocab_words),
    )

    # Get fallback bundle để fill các phần fail
    fallback = _fallback_vocab_bundle(vocab_words)

    # Post-processing và validation cuối cùng: vocab_story
    # Chấp nhận bất kỳ số đoạn >0, chỉ lọc đoạn rỗng; không reject để tránh mất story.
    if story:
        story_paras = story.get('paragraphs', [])
        if isinstance(story_paras, list):
            processed_paras = [p.strip() for p in story_paras if isinstance(p, str) and p.strip()]
            if processed_paras:
                story['paragraphs'] = processed_paras
                print(f"[summarizer] Final validation: vocab_story kept with {len(processed_paras)} paragraphs (very relaxed)")
            else:
                print(f"[summarizer] Final validation: vocab_story rejected (no usable paragraphs), using fallback")
                story = None
        else:
            print(f"[summarizer] Final validation: vocab_story rejected (invalid paragraphs type), using fallback")
            story = None
    
    # Cloze Tests: mỗi câu hỏi chỉ có 1 blank - POST-PROCESS để tách các blanks thành câu hỏi riêng
    safe_cloze_templates = {
        "have lunch": {
            "title": "Bữa trưa",
            "paragraph": "Tôi thường ăn ___1___ vào khoảng 12 giờ trưa.",
            "explanation": "Dùng 'have lunch' cho bữa trưa.",
            "example": "Chúng tôi thường ăn bữa trưa (have lunch) cùng đồng nghiệp."
        },
        "run": {
            "title": "Chạy bộ",
            "paragraph": "Mỗi sáng tôi ___1___ quanh công viên để khỏe mạnh.",
            "explanation": "Dùng 'run' cho hành động chạy.",
            "example": "Tôi thích chạy bộ (run) vào cuối tuần."
        },
        "walk": {
            "title": "Đi bộ",
            "paragraph": "Sau bữa tối, tôi thường ___1___ quanh khu phố.",
            "explanation": "Dùng 'walk' cho hành động đi bộ.",
            "example": "Chúng tôi đi bộ (walk) cùng nhau vào chiều mát."
        },
        "eat snack": {
            "title": "Ăn nhẹ",
            "paragraph": "Khi thấy đói giữa buổi, tôi sẽ ___1___ để nạp năng lượng.",
            "explanation": "Dùng 'eat snack' cho ăn nhẹ.",
            "example": "Tôi thích ăn nhẹ (eat snack) khi xem phim."
        },
        "drink water": {
            "title": "Uống nước",
            "paragraph": "Trong ngày, tôi luôn nhớ ___1___ để giữ sức khỏe.",
            "explanation": "Dùng 'drink water' cho hành động uống nước.",
            "example": "Bạn nên uống nước (drink water) đủ mỗi ngày."
        },
    }
    if cloze:
        valid_cloze = []
        import re
        for item in cloze:
            if not isinstance(item, dict):
                continue
            paragraph = item.get('paragraph', '')
            blanks = item.get('blanks', [])
            
            if not isinstance(blanks, list) or not blanks:
                continue
            
            blank_count_in_text = len(re.findall(r'___\d+___', paragraph))
            
            if blank_count_in_text == 1 and len(blanks) == 1:
                valid_cloze.append(item)
            elif blank_count_in_text > 1 or len(blanks) > 1:
                print(f"[summarizer] Post-processing cloze: splitting {blank_count_in_text} blanks into separate questions")
                for idx, blank in enumerate(blanks):
                    if not isinstance(blank, dict):
                        continue
                    blank_id = blank.get('id', idx + 1)
                    blank_answer = blank.get('answer', '')
                    
                    new_paragraph = paragraph
                    all_blank_matches = re.finditer(r'___(\d+)___', paragraph)
                    for match in all_blank_matches:
                        match_id = int(match.group(1))
                        if match_id != blank_id:
                            other_blank = next((b for b in blanks if b.get('id') == match_id), None)
                            replacement = other_blank.get('answer', '') if other_blank else ''
                            if replacement:
                                new_paragraph = new_paragraph.replace(f"___{match_id}___", replacement)
                    
                    remaining_blanks = len(re.findall(r'___\d+___', new_paragraph))
                    if remaining_blanks == 1 and blank_answer:
                        valid_cloze.append({
                            "title": item.get('title', f"Question {blank_id}"),
                            "paragraph": new_paragraph,
                            "blanks": [{"id": blank_id, "answer": blank_answer, "explanation": blank.get('explanation', ''), "on_correct_example": blank.get('on_correct_example', '')}]
                        })
            else:
                print(f"[summarizer] Final validation: cloze item rejected (no blanks found)")
        
        if valid_cloze:
            rebuilt = []
            for item in valid_cloze:
                blanks = item.get("blanks", [])
                if not blanks:
                    continue
                ans = blanks[0].get("answer", "")
                key = ans.lower().strip()
                tpl = safe_cloze_templates.get(key)
                if tpl:
                    rebuilt.append({
                        "title": tpl["title"],
                        "paragraph": tpl["paragraph"].replace("___1___", "___1___"),
                        "blanks": [{
                            "id": 1,
                            "answer": ans,
                            "explanation": tpl["explanation"],
                            "on_correct_example": tpl["example"]
                        }]
                    })
                else:
                    rebuilt.append(item)
            cloze = rebuilt
            print(f"[summarizer] Final validation: cloze_tests post-processed to {len(cloze)} questions")
        else:
            print(f"[summarizer] Final validation: all cloze_tests rejected, using fallback")
            cloze = None
    
    if match_pairs:
        valid_pairs = []
        placeholder_patterns = [
            "nghĩa của", "nghĩa ngắn gọn", "ý nghĩa ngắn gọn", 
            "thực tế của", "meaning of", "nghĩa của từ",
            "nghĩa cụ thể của", "nghĩa tiếng việt của", "nghĩa của x"
        ]
        
        translation_map = {}
        if summary_table:
            for row in summary_table:
                if isinstance(row, dict):
                    word = row.get('word', '').lower()
                    translation = row.get('translation', '')
                    if word and translation:
                        translation_map[word] = translation
        
        seen_words = set()
        for item in match_pairs:
            if not isinstance(item, dict):
                continue
            word = item.get('word', '')
            meaning = item.get('meaning', '')
            meaning_lower = str(meaning).lower()
            
            is_placeholder = any(pattern in meaning_lower for pattern in placeholder_patterns)
            
            word_key = word.lower().strip()
            if not word_key or word_key in seen_words:
                continue
            seen_words.add(word_key)

            if is_placeholder or not meaning.strip():
                if word_key in translation_map:
                    item['meaning'] = translation_map[word_key]
                else:
                    continue
            meaning_clean = item.get('meaning', '').strip()
            if not meaning_clean:
                continue
            words_meaning = meaning_clean.split()
            meaning_short = " ".join(words_meaning[:3])
            if len(meaning_short) > 20:
                meaning_short = meaning_short[:20].rstrip()
            item['meaning'] = meaning_short
            valid_pairs.append(item)
        
        if summary_table:
            for row in summary_table:
                if not isinstance(row, dict):
                    continue
                w = row.get('word', '')
                if not w:
                    continue
                w_key = w.lower().strip()
                if any(p.get('word', '').lower().strip() == w_key for p in valid_pairs):
                    continue
                trans = row.get('translation', '')
                if not trans:
                    continue
                meaning_short = " ".join(trans.split()[:3])
                if len(meaning_short) > 20:
                    meaning_short = meaning_short[:20].rstrip()
                valid_pairs.append({"id": len(valid_pairs) + 1, "word": w, "meaning": meaning_short})

        if valid_pairs:
            match_pairs = valid_pairs  
            print(f"[summarizer] Final validation: match_pairs post-processed to {len(match_pairs)} pairs")
        else:
            print(f"[summarizer] Final validation: all match_pairs rejected, using fallback")
            match_pairs = None

    return {
        "summary_table": summary_table if summary_table else fallback.get("summary_table"),
        "vocab_story": story if story else fallback.get("vocab_story"),
        "vocab_mcqs": mcqs if mcqs else fallback.get("vocab_mcqs"),
        "flashcards": flashcards if flashcards else fallback.get("flashcards"),
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
