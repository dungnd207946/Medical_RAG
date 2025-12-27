# System Versioning and Configuration Management

This document describes how versioning is implemented in the Medical RAG System.
Instead of relying on external experiment-tracking tools, the system adopts a
lightweight, configuration-driven versioning strategy that is suitable for
research prototypes and early-stage production systems.

---

## 1. Overview

The Medical RAG System supports multiple system versions to enable controlled
experimentation and iterative improvements. Each version defines a complete
RAG pipeline configuration, including:

- Knowledge base version
- Embedding model
- Vector index configuration
- LLM and prompt version
- Retrieval strategy

At runtime, the active system version is selected via an environment variable
and dynamically loaded by the application.

---

## 2. Version Selection Mechanism

The system version is controlled by the environment variable:

```bash
RAG_VERSION=v1.0
