# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a Chatbot with Retrieval-Augmented Generation (RAG) system that combines
information retrieval with large language models to generate grounded and context-aware answers in medical field, focus on injury prevention.

The system is designed with a modular and containerized architecture for easy deployment and scalability.

---

## RAG Pipeline
1. User submits a query
2. The query is converted into an embedding vector
3. Relevant documents are retrieved from the vector database
4. Retrieved context is injected into the LLM prompt
5. The LLM generates the final response

---

## System Architecture
The system consists of the following components:
- **API Service**: Handles user requests and orchestrates the RAG pipeline
- **Embedding Service**: Generates embeddings for queries and documents
- **Vector Database**: Stores and retrieves document embeddings
- **LLM Inference Service**: Generates answers using retrieved context

Refer to the [Architecture](architecture.md) section for more details.

---

## Technology Stack
- Python, Flask
- HuggingFace Transformers
- Vector Database (FAISS)
- Docker & Docker Compose
- MkDocs (for documentation)

---

## Documentation Guide
- [Architecture](architecture.md): Overall system design
- [Data ](data_storage_evaluation.md): Document loading and indexing
- [System Versioning](versioning.md): Version setting
---

## Use Cases
- Question answering over private documents
- Knowledge base assistant
- Internal enterprise search

---

## Getting Started
To run the system locally:
```bash
docker compose up
