# 📚 RAG Classic – Production-Structured Retrieval-Augmented Generation Pipeline

A clean, modular, production-style implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built with FastAPI, vector search, reranking, and LLM-based answer generation with citations.

This project demonstrates how modern AI systems retrieve, rerank, and generate grounded answers from documents like financial reports.

---


## 📁 Folder & File Overview

### 🔹 `app/`
Core RAG pipeline logic.

- **config.py** → Configuration settings (API keys, model names, vector DB configs)
- **ingestion.py** → Document loading & chunking
- **embedding.py** → Embedding generation logic
- **retrieval.py** → Vector search & Top-K retrieval
- **reranker.py** → Re-ranking retrieved documents
- **generation.py** → LLM response generation
- **api.py** → FastAPI endpoints

---

### 🔹 `docs/`
Stores source documents to be ingested into the vector database.

---

### 🔹 Root Files

- **main.py** → Application entry point
- **pyproject.toml** → Dependency and project configuration
- **README.md** → Project documentation




