# Data Storage Evaluation

## Overview

This document evaluates the data storage and retrieval layer used in the Medical Chatbot using Retrieval-Augmented Generation (RAG) system.  

The system adopts a **dual-storage architecture**:
- **Elasticsearch** for lexical (keyword-based) retrieval
- **FAISS** for dense vector storage and semantic similarity search

This separation allows each system to focus on its core strengths, improving both retrieval quality and system performance.

---

## Data Storage Architecture

| Component | Responsibility                            |
|---------|-------------------------------------------|
| Elasticsearch | Full-text search using BM25               |
| FAISS | Dense vector storage and semantic search  |
| Embedding Model | bioBERT                                   |
| Storage Format | JSONL (documents), `.index` (FAISS index) |

---

## Elasticsearch: Full-Text Storage and Retrieval

### Internal Architecture

Elasticsearch is built on **Java** and uses the **Apache Lucene** search engine internally.

Key characteristics:
- Text is indexed using **inverted indexes**
- Optimized for keyword-based retrieval
- Uses **BM25 ranking algorithm**
- Scales efficiently for large document collections

In this project, Elasticsearch is **not used for vector storage**.

---

### Full-Text Indexing and Search

Elasticsearch is responsible for:
- Token-based text indexing
- Exact and fuzzy keyword matching
- Initial lexical filtering in hybrid retrieval

**Ranking algorithm**:
- BM25 (Okapi BM25)

This approach is particularly effective when:
- Queries contain domain-specific terminology
- Exact keyword matches are important
- Semantic embeddings may miss rare or technical terms

---

## FAISS: Vector Storage and Semantic Search

### Overview of FAISS

FAISS is a library optimized for:
- Efficient similarity search
- High-dimensional vector storage
- Approximate and exact nearest neighbor search

FAISS is implemented in **C++ with Python bindings** and supports:
- CPU and GPU acceleration
- Multiple indexing strategies

---

### Vector Index Configuration

- **Embedding model**: bioBERT
- **Vector dimension**: 768
- **Similarity metric**: Cosine similarity
- **Persistence format**: `faiss_index.index`

All document embeddings are stored locally in a FAISS index file, enabling fast semantic retrieval without external dependencies.

---

### Index Creation and Persistence

Steps performed:

1. Generated embeddings for document chunks
2. Added vectors to FAISS index
3. Persisted index to disk as `faiss_index.index`
4. Loaded index into memory during inference

This approach ensures:
- Fast startup after index loading
- Reproducibility across runs
- Clear versioning of vector indices

---

### Semantic Search Workflow

1. User query is encoded using the same embedding model
2. FAISS performs KNN search
3. Top-K most similar document vectors are returned
4. Corresponding text chunks are retrieved for RAG generation

FAISS enables:
- Low-latency semantic search
- Deterministic similarity scoring
- Efficient retrieval even for large embedding sets

---

## Data Ingestion and Indexing Performance

### Elasticsearch (Full-Text)

- Indexed text fields only (no vectors)
- Used bulk indexing for performance
- Stable indexing behavior with minimal I/O overhead

### FAISS (Vector Index)

- Index creation time depends on:
  - Number of documents
  - Vector dimension
  - Index type (Flat vs ANN)
- Once built, query latency is significantly lower than vector search in Elasticsearch

---

## Retrieval Strategy Comparison

### Lexical Retrieval (Elasticsearch)
- BM25 ranking
- Fast and precise keyword matching
- Sensitive to vocabulary mismatch

### Semantic Retrieval (FAISS)
- Dense embedding similarity
- Captures semantic meaning
- Robust to paraphrasing

### Hybrid Retrieval

The system combines both approaches:
1. Keyword-based filtering using Elasticsearch
2. Semantic reranking or retrieval using FAISS

This hybrid approach improves:
- Recall for complex queries
- Relevance in domain-specific contexts

---

## Scalability and Performance Considerations

### FAISS
- Extremely fast for single-node workloads
- Suitable for GPU acceleration
- Requires manual sharding for large-scale distributed setups

### Elasticsearch
- Scales horizontally across nodes
- Well-suited for metadata filtering and keyword search

The combination balances **scalability**, **speed**, and **retrieval quality**.

---

## Conclusion

The adopted dual-storage strategy leverages the strengths of both systems:

- **Elasticsearch** provides robust and scalable full-text search
- **FAISS** enables efficient and accurate semantic retrieval

This architecture is well-suited for RAG pipelines, offering high-quality context retrieval while maintaining system simplicity and performance.
