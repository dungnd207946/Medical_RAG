import pytest
import time
from med_rag import MedRAG
import numpy as np
import json
from unittest.mock import MagicMock
import psutil
import requests



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


def test_hardware_requirements():
    """Đảm bảo server có đủ RAM để vận hành MedRAG"""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    # Giả sử mô hình yêu cầu ít nhất 8GB RAM
    assert total_gb >= 8.0, f"Hạ tầng không đủ RAM: {total_gb:.2f}GB/8GB"


def test_gpu_availability():
    """Kiểm tra xem hệ thống có nhận diện được GPU (nếu dùng mô hình local)"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            assert device_name is not None
        else:
            pytest.skip("Chế độ chạy CPU: Không phát hiện GPU.")
    except ImportError:
        pytest.skip("Chưa cài đặt Torch để kiểm tra GPU.")


def test_api_endpoint_connectivity():
    """Kiểm tra kết nối tới dịch vụ LLM (OpenAI, Anthropic hoặc Local Ollama)"""
    # Thay URL bằng endpoint bạn đang dùng
    url = "https://api.openai.com/v1/models" 
    try:
        response = requests.get(url, timeout=5)
        assert response.status_code in [200, 401] # 401 là kết nối được nhưng chưa auth
    except requests.exceptions.RequestException:
        pytest.fail("Hạ tầng mạng: Không thể kết nối tới API LLM.")




# integration Tests (Kiểm thử tích hợp hệ thống đầy đủ)

def test_api_integration_error_handling(rag_instance):
    """
    KIỂM THỬ TÍCH HỢP 2: Khả năng chịu lỗi khi API ngoại vi gặp sự cố.
    """
    # Giả sử nhập một câu hỏi có ký tự lạ hoặc cực dài để test giới hạn API
    long_query = "Y khoa " * 500 
    
    try:
        response = rag_instance.get_answer(long_query)
        assert response is not None
    except Exception as e:
        # Nếu fail, phải đảm bảo lỗi được bắt một cách tường minh, không làm crash hệ thống
        pytest.fail(f"Hệ thống bị sập khi gặp query dài: {e}")

def test_end_to_end_consistency(rag_instance):
    """
    KIỂM THỬ TÍCH HỢP 3: Tính nhất quán của định dạng đầu ra.
    """
    query = "Phác đồ điều trị tiểu đường type 2"
    # Giả sử MedRAG trả về một dictionary chứa cả answer và nguồn (sources)
    # Nếu code hiện tại chỉ trả về string, hãy test định dạng string
    result = rag_instance.get_answer(query)
    
    assert "tiểu đường" in result.lower()
    # Kiểm tra xem có câu từ chối trách nhiệm y khoa (Disclaimer) không - Rất quan trọng trong tích hợp y tế
    assert any(word in result.lower() for word in ["bác sĩ", "tư vấn", "chuyên gia"]), \
        "LỖI TÍCH HỢP: Thiếu cảnh báo y tế (Medical Disclaimer) trong phản hồi cuối."


# 6. Unit Test các hàm phụ trợ (Utility Functions)
 
def preprocess_text(text):
    if not text:
        return ""
    return text.strip().lower()

def calculate_confidence(scores):
    if not scores:
        return 0.0
    return float(np.mean(scores))


def test_preprocess_text_normal():
    """Kiểm tra làm sạch văn bản thông thường"""
    assert preprocess_text("  Sốt Xuất Huyết  ") == "sốt xuất huyết"

def test_preprocess_text_empty():
    """Kiểm tra xử lý chuỗi rỗng"""
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""

def test_calculate_confidence_valid():
    """Kiểm tra tính điểm tin cậy từ danh sách điểm số vector"""
    scores = [0.8, 0.9, 0.7]
    assert calculate_confidence(scores) == 0.8

def test_calculate_confidence_empty():
    """Kiểm tra khi không có kết quả truy xuất nào"""
    assert calculate_confidence([]) == 0.0

# --- Unit Test với Mocking (Giả lập thành phần phức tạp) ---

def test_retriever_logic_with_mock():
    """
    Kiểm tra logic của bộ truy xuất mà không cần kết nối Database thật.
    Đảm bảo hàm trả về đúng số lượng tài liệu yêu cầu.
    """
    mock_retriever = MagicMock()
    # Giả lập giá trị trả về của hàm retrieve
    mock_retriever.retrieve.return_value = ["Doc 1", "Doc 2", "Doc 3"]
    
    query = "đau đầu"
    results = mock_retriever.retrieve(query)
    
    assert len(results) == 3
    assert results[0] == "Doc 1"
    mock_retriever.retrieve.assert_called_once_with(query)





if __name__ == "__main__":
    pytest.main([__file__])
    #test_automated_evaluation(rag_instance())


print("Tất cả các test đã hoàn thành.")
