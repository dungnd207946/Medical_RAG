import json
import time
from openAI_chat import Chat, config_version
from bioBERT_retriever import BioBERTRetriever
from bm25_retriever import BM25Retriever
from hybrid_retriever import HybridRetriever
# from medCPT_retriever import MedCPTRetriever

class MedRAG:
    def __init__(self, retriever=1, question_type=1, n_docs=config_version["retrieval"]["top_k"]):
        if retriever == 1:
            self.retriever = BioBERTRetriever()
        elif retriever == 2:
            self.retriever = BM25Retriever()
        elif retriever == 3:
            self.retriever = HybridRetriever()
        # elif retriever == 4:
        #     self.retriever = MedCPTRetriever(rerank=True)
        else:
            raise ValueError("Invalid retriever value. Choose 1 for bioBERT, 2 for BM25, or 3 for hybrid.")

        self.chat = Chat(question_type=question_type) # 1 for full text, 2 for yes/no
        self.n_docs = n_docs

    def get_answer(self, question: str) -> str:

        # retrieve the documents timing the retrieval
        start_time_retrieval = time.time()
        retrieved_docs = json.loads(self.retriever.retrieve_docs(question, self.n_docs))

        print("Retrieved docs:", retrieved_docs)
        end_time_retrieval = time.time()

        start_time_generation = time.time()
        answer = self.chat.create_chat(question, retrieved_docs)
        end_time_generation = time.time()

        retrieval_time = end_time_retrieval - start_time_retrieval
        generation_time = end_time_generation - start_time_generation

        return answer