from elasticsearch import Elasticsearch
import os
import requests
import json
from medical_RAG_system.information_retrieval.document_encoding.bioBERT_encoder import bioBERTEncoder
from medical_RAG_system.information_retrieval.faiss_container.faiss_insert_data import FaissData
from dotenv import dotenv_values
import os
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
FAISS_URL   = os.getenv("FAISS_URL",   "http://localhost:5000/search")

class BioBERTRetriever:
    def __init__(self):
        # self.elastic_password = "7n2xK2kELC0GYsOCyi9+"
        # ca_certs = r"C:\Users\Dung\Downloads\elasticsearch-9.2.0-windows-x86_64\elasticsearch-9.2.0\config\certs\http_ca.crt"
        self.es = Elasticsearch([ELASTIC_URL], request_timeout=60)
        self.index = "injury_prevent_index"
        self.faiss_url = FAISS_URL
        self.query_encoder = bioBERTEncoder()

    def query_to_vector(self, text: str):
        """ Biến đổi query của người dùng thành vector sử dụng BioBERT encoder."""
        embedding = self.query_encoder.encode(text)
        return embedding

    def faiss_query(self, query: str, k: int = 5):
        """Performs a vector search using FAISS with the given query and k."""
        faiss_data = FaissData()
        vec = self.query_to_vector(query).tolist()  # Convert numpy array to list
        data = {
            'queries': [vec],  # List of vectors
            'k': k
        }
        response = requests.post(self.faiss_url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))
        return response.json()

    def get_docs_via_IDs(self, IDs: list):
        body = {
            "size": len(IDs),
            "query": {
                "terms": {
                    "id": IDs
                }
            },
            "_source": ["id", "title", "text_chunked"]
        }
        return self.es.search(index=self.index, body=body)

    def retrieve_docs(self, query: str, k: int = 5):
        """ Truy vấn các tài liệu liên quan sử dụng cả FAISS và Elasticsearch."""
        response = self.faiss_query(query, k)
        IDs = response['ids'][0]  # Assumes PMIDs are returned in a structured list
        es_response = self.get_docs_via_IDs(IDs)
        # Có thể thêm re-rank vào đây
        results = {}

        # Formatting the response as required
        for idx, hit in enumerate(es_response['hits']['hits'], 1):
            doc_key = f"doc{idx}"
            results[doc_key] = {
                'id': hit['_source']['id'],
                'title': hit['_source']['title'],
                'text_chunked': hit['_source']['text_chunked']
            }

        return json.dumps(results, indent=4)

if __name__ == '__main__':
    retriever = BioBERTRetriever()
    query = "Why injury prevention in sports important?"
    n_docs = 5
    response = retriever.retrieve_docs(query, n_docs)
    print(response)