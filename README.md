# Optima - Backend

Decision Support System backend for a fashion retail store. Combines:

- XGBoost regression for weekly sales prediction
- What-if simulation (price increase, discount change, extended discount campaign)
- RAG-powered chatbot (OpenAI GPT-4o-mini + ChromaDB + SentenceTransformers)
- FastAPI REST API consumed by the optima-frontend React dashboard

---

## File overview
```

Optima/
├── api.py              FastAPI endpoints 
├── chatbot.py          LLM router + what-if & RAG handlers
├── knowledge_base.py   Doc generation, ChromaDB setup, retrieval
├── whatif.py           Scenario simulation engine
├── data_loader.py      Loads training CSV + pickled XGBoost model
├── config.py           Constants (product IDs, paths, valid discounts, ...)
├── .env.example        fill OPENAI_API_KEY
└── .gitignore
```

---


## Architecture
```


User question
    │
    ▼
POST /chat → chatbot.classify_user_query() (LLM)
    │
    ├── route = "what_if"       → parse params (LLM) → whatif.* → XGBoost predict
    │                                               → generate_whatif_llm_response (LLM)
    │
    ├── route = "statistics_rag" → ChromaDB filtered retrieval
    │                            → answer_statistics_rag_question (LLM)
    │
    └── route = "unknown"        → polite fallback message

```

Built as a graduation project - 2026.
