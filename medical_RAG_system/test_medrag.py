import pytest
import time
from med_rag import MedRAG


print("Bắt đầu chạy các test cho MedRAG...")
# Fixture để khởi tạo mô hình một lần cho tất cả các test

@pytest.fixture(scope="module")
def rag_instance():
    return MedRAG(retriever=1, question_type=1)

# 1. Kiểm tra khởi tạo (Smoke Test)
def test_initialization(rag_instance):
    assert rag_instance is not None
    assert hasattr(rag_instance, 'get_answer'), "Mô hình thiếu hàm get_answer"

# 2. Kiểm tra API Response (Định dạng kết quả)
def test_api_response_format(rag_instance):
    question = "Bệnh cúm A có triệu chứng gì?"
    answer = rag_instance.get_answer(question)
    
    assert isinstance(answer, str), "Kết quả trả về phải là một chuỗi (string)"
    assert len(answer) > 0, "Kết quả trả về không được để trống"

# 3. Kiểm tra tốc độ phản hồi (Performance Test)
def test_api_latency(rag_instance):
    question = "Định nghĩa về cao huyết áp?"
    start_time = time.time()
    rag_instance.get_answer(question)
    latency = time.time() - start_time
    
    # Giới hạn 10 giây cho một phản hồi y tế phức tạp
    assert latency < 10, f"API phản hồi quá chậm: {latency:.2f}s"

# 4. Kiểm tra khả năng Chunking/Retrieval (Logic Test)
def test_retrieval_logic(rag_instance):
    # Test xem retriever có lấy được nội dung liên quan không
    # Giả sử MedRAG có thuộc tính retriever
    query = "thuốc paracetamol"
    if hasattr(rag_instance, 'retriever'):
        docs = rag_instance.retriever.retrieve(query)
        assert len(docs) > 0, "Retriever không tìm thấy tài liệu nào cho 'paracetamol'"
        # Kiểm tra xem từ khóa có xuất hiện trong các chunk đầu tiên không
        content = docs[0].page_content.lower()
        assert "paracetamol" in content or "thuốc" in content

# 5. Kiểm tra xử lý câu hỏi rỗng hoặc không hợp lệ
def test_empty_query(rag_instance):
    with pytest.raises(Exception): # Hoặc kiểm tra cách code bạn xử lý chuỗi rỗng
        rag_instance.get_answer("")

if __name__ == "__main__":
    pytest.main([__file__])


print("Tất cả các test đã hoàn thành.")