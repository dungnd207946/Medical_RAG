import pytest
import time
from med_rag import MedRAG
import numpy as np
import json



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

def test_embedding_semantic_similarity(rag_instance):
    # Giả sử rag_instance có thuộc tính embedding_model
    if hasattr(rag_instance, 'embedding_model'):
        text1 = "Bệnh nhân bị đau đầu kinh niên"
        text2 = "Triệu chứng đau đầu kéo dài"
        text3 = "Cách nấu món phở bò"
        
        vec1 = np.array(rag_instance.embedding_model.embed(text1))
        vec2 = np.array(rag_instance.embedding_model.embed(text2))
        vec3 = np.array(rag_instance.embedding_model.embed(text3))
        
        # Tính Cosine Similarity
        sim12 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        sim13 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        
        # Kỳ vọng: Câu y khoa cùng nghĩa phải có độ tương đồng cao hơn câu lạc đề
        assert sim12 > sim13, f"Embedding không nhận diện được sự tương đồng y khoa: {sim12} vs {sim13}"
        assert sim12 > 0.7, "Độ tương đồng ngữ nghĩa quá thấp"

def test_inference_reasoning(rag_instance):
    # Kiểm tra khả năng suy luận dựa trên ngữ cảnh giả lập
    context = "Bệnh nhân A dị ứng với Penicillin. Bác sĩ kê đơn Amoxicillin (một loại thuộc nhóm Penicillin)."
    query = f"Dựa trên ngữ cảnh: {context}. Việc kê đơn này có an toàn không? Tại sao?"
    
    answer = rag_instance.get_answer(query).lower()
    
    # Kỳ vọng: LLM phải suy luận được sự nguy hiểm dựa trên mối quan hệ nhóm thuốc
    keywords = ["không an toàn", "nguy hiểm", "dị ứng", "penicillin"]
    assert any(word in answer for word in keywords), "LLM không thực hiện được suy luận logic y khoa"


def test_automated_evaluation(rag_instance):
    with open('D:/GitHub/Medical_RAG/medical_RAG_system/rag_system/test_medrag.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for case in data['samples']:
        answer = rag_instance.get_answer(case['question']).lower()
        # Tính điểm dựa trên số lượng keyword xuất hiện trong answer
        found = [k for k in case['keywords'] if k.lower() in answer]
        score = len(found) / len(case['keywords'])
        results.append(score)
    
    avg_score = sum(results) / len(results)
    print(f"\n[EVALUATION] Average Recall Score: {avg_score:.2f}")
    assert avg_score > 0.9, "Chất lượng tổng thể của mô hình chưa đạt yêu cầu y khoa cơ bản."





if __name__ == "__main__":
    pytest.main([__file__])
    #test_automated_evaluation(rag_instance())


print("Tất cả các test đã hoàn thành.")

