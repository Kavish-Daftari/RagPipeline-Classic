rag-classic/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Settings & environment variables
│   ├── ingestion.py       # PDF/TXT loading, page extraction, chunking
│   ├── embedding.py       # Create Pinecone index & upsert records
│   ├── retrieval.py       # Semantic vector search
│   ├── reranker.py        # Rerank with bge-reranker-v2-m3
│   ├── generation.py      # LLM answer generation with citations
│   └── api.py             # FastAPI REST endpoints
├── docs/                  # Source documents
│   ├── Apple_Q24.pdf
│   └── Nike-Inc-2025_10K.pdf
├── main.py                # CLI entry point (ingest / ask / serve)
├── pyproject.toml         # Dependencies
├── .env                   # API keys (not committed — see .gitignore)
├── .gitignore
└── README.md
