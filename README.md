# Optima — Backend (FastAPI)

Decision Support System backend for a fashion retail chain. Combines:

- **XGBoost regression** for weekly sales prediction
- **What-if simulation** (price increase, discount change, extended discount campaign)
- **RAG-powered chatbot** (OpenAI GPT-4o-mini + ChromaDB + SentenceTransformers)
- **FastAPI** REST API consumed by the [`optima-frontend`](https://github.com/AryafAI/optima-frontend) React dashboard

---

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET`  | `/`  | — | `{message: "Optima DSS API is running"}` |
| `GET`  | `/products` | — | `[{id, name}, …]` |
| `POST` | `/whatif/price` | `{product_id, price_increase}` | scenario result dict |
| `POST` | `/whatif/discount` | `{product_id, new_discount}` | scenario result dict |
| `POST` | `/whatif/extended` | `{product_id, start_month, end_month, new_discount}` | `{monthly_detail, total, status, message}` |
| `POST` | `/chat` | `{message}` | `{text, result, route}` |

CORS is open (`allow_origins=["*"]`) for development convenience.

---

## File overview

```
Optima/
├── api.py              ← FastAPI endpoints (entry point)
├── chatbot.py          ← LLM router + what-if & RAG handlers
├── knowledge_base.py   ← Doc generation, ChromaDB setup, retrieval
├── whatif.py           ← Scenario simulation engine
├── data_loader.py      ← Loads training CSV + pickled XGBoost model
├── config.py           ← Constants (product IDs, paths, valid discounts, …)
├── .env.example        ← Template — copy to .env and fill OPENAI_API_KEY
└── .gitignore
```

---

## Run locally

### 1. Clone

```powershell
cd $env:USERPROFILE\Documents
git clone https://github.com/AryafAI/Optima.git
cd Optima
```

### 2. Install dependencies

```powershell
pip install fastapi uvicorn pandas xgboost openai chromadb sentence-transformers python-dotenv pydantic
```

### 3. Create your `.env`

```powershell
copy .env.example .env
```

Open `.env` and replace the placeholder with your real OpenAI API key
(get one at https://platform.openai.com/api-keys).

### 4. Make sure the data files are reachable

`config.py` expects training data and the trained XGBoost model at:

```
G:\My Drive\Optima\Final_clean_data\final_train_data.csv
G:\My Drive\Optima\Optima_model\optima_xgb_model.pkl
G:\My Drive\Optima\Chatbot\optima_llm_rag_documents.json   (auto-generated if missing)
```

If your Google Drive isn't on `G:`, edit the three path constants at the bottom of `config.py`.

### 5. Run

```powershell
uvicorn api:app --reload
```

Server starts on http://localhost:8000

On first run, the RAG knowledge base is generated automatically (~30s, ~16 LLM calls)
and saved as `optima_llm_rag_documents.json` for future fast restarts.

---

## Frontend

The React dashboard that consumes this API:
[github.com/AryafAI/optima-frontend](https://github.com/AryafAI/optima-frontend)

Run both together:

```powershell
# Terminal 1
cd Optima
uvicorn api:app --reload

# Terminal 2
cd optima-frontend
npm install
npm run dev
```

The dashboard shows a live "Live API" badge at the top-right confirming the connection.

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

Built as a graduation project — Aryaf, May 2026.
