import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.append('d:/GitHub/Medical_RAG/medical_RAG_system')

from information_retrieval.data_process import DataProcess  # Thay your_module bằng tên file chứa class DataProcess


@pytest.fixture
def setup_test_env(tmp_path):
    # Tạo cấu trúc thư mục giả lập
    pdf_dir = tmp_path / "data" / "pdf_document"
    pdf_dir.mkdir(parents=True)
    
    # Tạo một file PDF giả (chỉ là file trống để glob tìm thấy)
    (pdf_dir / "test_doc.pdf").write_text("dummy content")
    
    # Tạo thư mục source và target giả
    Path("source").mkdir(exist_ok=True)
    Path("target").mkdir(exist_ok=True)
    
    return pdf_dir

class TestDataProcess:
    
    @patch('document_encoding.text_chunking.TextChunking')
    @patch('elastic_container.elastic_indexing.ElasticIndexing')
    @patch('document_encoding.bioBERT_encoder.bioBERTEncoder')
    @patch('faiss_container.faiss_insert_data.FaissData')
    def test_process_flow(self, mock_faiss, mock_bert, mock_elastic, mock_chunking, setup_test_env):
        """
        Kiểm tra xem tất cả các bước trong quy trình process() có được gọi đúng thứ tự và tham số không.
        """
        # Khởi tạo class với path giả lập
        dp = DataProcess()
        dp.pdf_document_path = setup_test_env
        
        # Thực thi quy trình
        dp.process()
        
        # 1. Kiểm tra bước Chunking
        mock_chunking.return_value.pdf_chungking.assert_called_once()
        args, kwargs = mock_chunking.return_value.pdf_chungking.call_args
        assert "medical_RAG_system\data\pdf_document\Sport_injury_prevent_2009.pdf" in args[0][0] # Kiểm tra xem đường dẫn file có đúng không

        # 2. Kiểm tra bước Elastic Indexing
        mock_elastic.assert_called_once_with("med_index")
        mock_elastic.return_value.indexing_documents.assert_called_once()

        # 3. Kiểm tra bước Encoding (BioBERT)
        mock_bert.return_value.embed_file.assert_called_once()

        # 4. Kiểm tra bước FAISS
        mock_faiss.return_value.insert_data.assert_called_once()

    def test_data_loss_after_chunking(self):
        """
        Kiểm tra xem dữ liệu sau khi chunking (file jsonl) có bị thiếu hụt nội dung không.
        """
        source_file = Path('medical_RAG_system/data/embed_data/source/text_chunked.jsonl')
        
        if not source_file.exists():
            pytest.skip("File source/text_chunked.jsonl chưa tồn tại, bỏ qua test dữ liệu.")

        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        assert len(lines) > 0, "Lỗi: File chunked trống, trích xuất PDF thất bại hoặc thiếu dữ liệu."
        
        # Kiểm tra cấu trúc một dòng dữ liệu
        first_line = json.loads(lines[0])
        assert "content" in first_line, "Mỗi bản ghi phải có trường 'content'."
        assert len(first_line["content"]) > 0, "Nội dung trích xuất không được để trống."

    
