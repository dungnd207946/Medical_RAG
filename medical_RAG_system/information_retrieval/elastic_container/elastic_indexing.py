#!/usr/bin/env python

import json
import os
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

elastic_url = os.getenv("ELASTIC_URL", "http://localhost:9200")
print("Using Elasticsearch:", elastic_url)   # thêm dòng này để debug

class ElasticIndexing:
    def __init__(self, index_name):
        #self.password = "7n2xK2kELC0GYsOCyi9+" # Mật khẩu elasticsearch trong máy bạn
        #ca_certs = r"C:\Users\Dung\Downloads\elasticsearch-9.2.0-windows-x86_64\elasticsearch-9.2.0\config\certs\http_ca.crt"
        
        self.es = Elasticsearch([elastic_url], request_timeout=60)
        self.index_name = index_name
        self.source_directory = Path(os.getenv("SOURCE_DIR", "/app/data/embed_data/source"))
        self.error_log_path = Path(os.getenv("ERROR_LOG_PATH", "/app/data/embed_data/errors.jsonl"))

    def create_index(self):
        if not self.es.indices.exists(index=self.index_name):
            # Định nghĩa mapping
            mapping = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "custom_lemmatizer_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stopwords", "porter_stem"]
                            }
                        },
                        "filter": {
                            "porter_stem": {
                                "type": "stemmer",
                                "language": "English"  # Specify the language for lemmatization
                            },
                            "stopwords": {
                                "type": "stop",
                                "stopwords": "_english_"  # the built-in English stop words list
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text_chunked": {
                            "type": "text",
                            "analyzer": "custom_lemmatizer_analyzer"
                        },
                        "title": {
                            "type": "keyword"  # title thường ngắn, dùng keyword để filter hoặc sort
                        },
                        "id": {
                            "type": "keyword"  # id dùng để lookup
                        }
                    }
                }
            }
            # Tạo mới index với mapping tương ứng
            self.es.indices.create(index=self.index_name, body=mapping)

    def indexing_documents(self):
        self.create_index()
        if not self.source_directory.exists():
            print("The source directory does not exist.")
            return

        actions = []  # List để lưu các document vào index

        # Open the error log file for writing
        with self.error_log_path.open('w') as error_log:
            # Lặp qua từng file trong folder source
            for file_name in tqdm(list(os.listdir(self.source_directory))):
                if file_name.endswith('.jsonl'):
                    source_file = self.source_directory / file_name

                    # Open and read the JSONL file
                    with open(source_file, 'r', encoding='utf-8') as json_file:
                        for line in json_file:
                            try:
                                doc = json.loads(line)

                                # Remove the "embeddings" field from the document
                                if "embeddings" in doc:
                                    del doc["embeddings"]

                                action = {
                                    "_index": self.index_name,
                                    "_source": doc
                                }
                                actions.append(action)

                                if len(actions) == 200:  # Bulk indexing threshold
                                    helpers.bulk(self.es, actions, raise_on_error=False)
                                    actions = []
                            except json.JSONDecodeError as e:
                                error_log.write(f"Error in file {file_name}: {e}\n")
                                error_log.write(f"{line}\n")
                            except Exception as e:
                                error_log.write(f"Unexpected error in file {file_name}: {e}\n")
                                error_log.write(f"{line}\n")

            # Index any remaining documents
            if actions:
                helpers.bulk(self.es, actions)

if __name__ == "__main__":
    # ca_certs=r"C:\Users\Dung\Downloads\elasticsearch-9.2.0-windows-x86_64\elasticsearch-9.2.0\config\certs\http_ca.crt"
    index_name = "injury_prevent_index"
    elastic_indexing = ElasticIndexing(index_name)
    elastic_indexing.indexing_documents()