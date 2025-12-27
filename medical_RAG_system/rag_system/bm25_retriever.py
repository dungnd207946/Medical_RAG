from elasticsearch import Elasticsearch
import os
import json

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")

class BM25Retriever:
    def __init__(self):
        self.es = Elasticsearch([ELASTIC_URL], request_timeout=60)
        self.index = "injury_prevent_index"

    def retrieve_docs(self, query: str, k: int = 10):
        es_query = {
            "size": k,
            "query": {
                "match": {
                    "content": query 
                }
            },
            "_source": ["id", "title", "text_chunked"]
        }
        # Execute the search query
        response = self.es.search(index=self.index, body=es_query)
        
        # Format the results into the desired JSON structure
        results = {}
        for idx, hit in enumerate(response['hits']['hits'], 1):
            doc_key = f"doc{idx}"
            results[doc_key] = {
                'id': hit['_source']['id'],
                'title': hit['_source']['title'],
                'text_chunked': hit['_source']['text_chunked']
            }

        return json.dumps(results, indent=4)