# System Architecture

## 1. Overview

This document describes the architecture of the chatbot using Retrieval-Augmented Generation (RAG)
system. The system is designed to provide accurate, context-aware answers by combining information retrieval techniques with
large language models (LLMs).


---

## 2. High-Level Architecture

At a high level, the system follows a standard RAG pipeline:

User Query  
→ Retriever  
→ Relevant Documents  
→ Prompt Construction  
→ Large Language Model (LLM)  
→ Generated Response

The system is composed of independent components that can be updated or replaced
without affecting the entire pipeline.

---

## 3. Core Components

### 3.1 User Interface / API Layer

**Description:**  
The system exposes a RESTful API that receives user queries and returns generated
responses.

**Implementation:**
- Framework: Flask, Streamlit
- Endpoint example: `/search`
- Input: User query (text)
- Output: Generated answer (text)

---

### 3.2 Document Ingestion Pipeline

**Description:**  
This component is responsible for preparing raw documents for retrieval.

**Steps:**
1. Load raw documents from the data source
2. Text cleaning and normalization
3. Document chunking
4. Embedding generation
5. Storage in vector database

**Configuration Parameters:**
- Chunk size: 10 sentences each chunk
- Embedding model: bioBERT

---

### 3.3 Embedding Model

**Description:**  
The embedding model transforms text chunks into dense vector representations used
for semantic search.

**Implementation:**
- Model: BioBERT
- Embedding dimension: 768
- Inference mode: Local / API-based

---

### 3.4 Retrieval Layer

**Description:**  
The retrieval layer identifies relevant documents based on the user query.

**Supported Retrieval Strategies:**
- Keyword-based retrieval (BM25)
- Semantic retrieval using vector similarity
- Hybrid retrieval combining BM25 and semantic search

**Backend Options:**
- Vector database: FAISS
- Similarity metric: cosine similarity

---

### 3.5 Prompt Construction

**Description:**  
Retrieved documents are injected into a predefined prompt template to guide the
LLM during response generation.

**Prompt Strategy:**
- Prompt versioning supported
- Context window limited to top-k retrieved documents

**Prompt Location:**
- `rag_system/openAI_chat/`

---

### 3.6 Large Language Model (LLM)

**Description:**  
The LLM generates the final response based on the constructed prompt.

**Implementation:**
- Provider: Google Gemini
- Model version: gemini-2.5-flash

---

## 4. Data Flow

1. User submits a query via the API
2. Query is processed by the retriever
3. Relevant documents are fetched from the vector store
4. Documents are injected into the prompt template
5. Prompt is sent to the LLM
6. Generated response is returned to the user

---

## 5. Deployment Architecture

**Environment:**
- Development: Local machine
- Deployment: Docker container, ...

**Containerization:**
- Docker used to ensure reproducibility
- Environment variables used for API keys and configuration

---

## 6. Design Principles

The system architecture follows these principles:

- **Modularity:** Each component can be modified independently
- **Reproducibility:** Docker and configuration files ensure consistent behavior
- **Scalability:** Retrieval and generation layers can be scaled separately
- **Observability:** Logging and evaluation hooks are supported

---

## 7. Limitations and Future Improvements

**Current Limitations:**
- No online feedback loop
- Manual re-indexing required for new data
- Limited automated evaluation

**Future Work:**
- Continuous indexing pipeline
- Online monitoring and alerting
- A/B testing for LLM and retrieval strategies
