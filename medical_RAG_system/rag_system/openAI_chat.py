import openai
import google.generativeai as genai
import os
import json
import yaml
from typing import List, Dict
from dotenv import dotenv_values, load_dotenv

load_dotenv("medical_RAG_system/pass.env")
version = os.getenv("RAG_VERSION")
with open(f"medical_RAG_system/versions/{version}.yaml") as f:
    config = yaml.safe_load(f)

config_version = config

class Chat:
    def __init__(self, question_type: int = 1, model=genai.GenerativeModel("models/gemini-2.5-flash")):
        self.api_key = dotenv_values("medical_RAG_system/pass.env")["HF_KEY"]
        genai.configure(api_key=self.api_key)
        self.model = model
        self.context = self.set_context(question_type)

    def set_context(self, question_type: int) -> str:
        base_context = (
            "You are a scientific assistant designed to synthesize responses "
            "from specific documents. Using the information provided in the "
            "documents and your base knowledge to answer questions. The first documents should be the most relevant."
            "Remember, documents accuracy is always a priority."
            "When answering questions, always apply knowledge from the doc to make inferences, answer naturally like human."
            "Please think step-by-step before answering questions and provide the most accurate response possible."
        )

        question_specific_context = {
            1: " Provide a detailed answer to the question in the 'response' field.",
            2: " Your response should only be 'yes', 'no'. If if no relevant documents are found, return 'no_docs_found'.",
            3: " Choose between the given options 1 to 4 and return as 'response' the chosen number. If no relevant documents are found, return the number 5.",
            4: " Respond with keywords and list each keyword sepeartly as a list element. For example ['keyword1', 'keyword2', 'keyword3']. If no relevant documents are found, return an empty list.",
        }

        return base_context + question_specific_context.get(question_type, "")

    def set_initial_message(self) -> List[dict]:
        return [{"role": "system", "content": self.context}]

    def create_chat(self, user_message: str, retrieved_documents: Dict) -> str:
        messages = self.set_initial_message()
        messages.append({"role": "user", "content": f"Answer the following question: {user_message}"})
        
        document_texts = ["id {}: {} {}".format(doc['id'], doc['title'], doc['text_chunked']) for doc in retrieved_documents.values()]
        documents_message = "\n\n".join(document_texts)  # Separating documents with two newlines
        messages.append({"role": "system", "content": documents_message})

        # 🧠 Tạo prompt tổng hợp cho Gemini
        prompt = (
            f"{self.context}\n\n"
            f"User question: {user_message}\n\n"
            f"Here are the retrieved documents:\n{documents_message}\n\n"
            "Please answer naturally, like a human medical expert. "
            "Use information from the retrieved documents to answer questions. "
            "If the documents are irrelevant, politely explain that you cannot answer. "
            "If the question is not relevant to academic, just normal question, please answer generally like other chatbot."
            "Do NOT output JSON or any structured data — only write the answer naturally."
            "The language of the answer must base on the language of user question."
        )

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            elif response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content.parts and len(candidate.content.parts) > 0:
                    return candidate.content.parts[0].text.strip()
                else:
                    return "⚠️ Mô hình không tạo ra phần nội dung hợp lệ."
            else:
                return "⚠️ Không có phản hồi từ mô hình."
            # return response_content
        except Exception as e:
            # return json.dumps({"error": str(e)})
            return f"Lỗi khi gọi mô hình: {str(e)}"