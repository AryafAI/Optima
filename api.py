# api.py - FastAPI endpoints for the Optima DSS

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from config import FIXED_STORE_ID, PRODUCT_NAMES, PRODUCT_IDS, VALID_DISCOUNTS
from data_loader import load_all
from whatif import (
    get_latest_baseline,
    get_baseline,
    whatif_price_change,
    whatif_avg_discount,
    whatif_extended_discount
)
from knowledge_base import (
    setup_chromadb,
    load_documents,
    generate_documents_with_llm
)
from chatbot import create_llm_client, call_llm, chat

# Startup - load everything
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Optima DSS API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve the interface folder as static files
app.mount("/interface", StaticFiles(directory="interface"), name="interface")

# Load data, model, LLM client, and ChromaDB on startup
train_data, model, product_avg_price, pr_min, pr_max = load_all()
llm_client = create_llm_client(OPENAI_API_KEY)

# Load or generate RAG documents
try:
    documents, metadatas, ids = load_documents()
    print("✓ Loaded documents from backup")
except FileNotFoundError:
    print("No backup found — generating documents...")
    documents, metadatas, ids = generate_documents_with_llm(
        train_data   = train_data,
        store_id     = FIXED_STORE_ID,
        product_ids  = PRODUCT_IDS,
        call_llm     = lambda p: call_llm(llm_client, p)
    )

collection = setup_chromadb(documents, metadatas, ids)

# Request Models

class PriceChangeRequest(BaseModel):
    product_id:        int
    price_increase:    float

class DiscountChangeRequest(BaseModel):
    product_id:  int
    new_discount: float

class ExtendedDiscountRequest(BaseModel):
    product_id:   int
    start_month:  int
    end_month:    int
    new_discount: float

class ChatRequest(BaseModel):
    message: str

# Endpoints

@app.get("/")
def root():
    return {"message": "Optima DSS API is running"}


@app.get("/products")
def get_products():
    """Returns list of available products for the dropdown."""
    return [
        {"id": pid, "name": PRODUCT_NAMES[pid]}
        for pid in PRODUCT_IDS
    ]


@app.post("/whatif/price")
def price_change(req: PriceChangeRequest):
    """Simulates a price increase for a product."""
    try:
        baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, req.product_id)
        result = whatif_price_change(baseline, baseline_pred, req.price_increase)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whatif/discount")
def discount_change(req: DiscountChangeRequest):
    """Simulates a discount change for a product."""
    if req.new_discount not in VALID_DISCOUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"Discount must be one of: {VALID_DISCOUNTS}"
        )
    try:
        baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, req.product_id)
        result = whatif_avg_discount(baseline, baseline_pred, req.new_discount)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whatif/extended")
def extended_discount(req: ExtendedDiscountRequest):
    """Simulates an extended discount campaign across multiple months."""
    if req.new_discount not in VALID_DISCOUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"Discount must be one of: {VALID_DISCOUNTS}"
        )
    if req.start_month > req.end_month:
        raise HTTPException(
            status_code=400,
            detail="start_month must be less than or equal to end_month."
        )
    try:
        result = whatif_extended_discount(
            store_id    = FIXED_STORE_ID,
            product_id  = req.product_id,
            start_month = req.start_month,
            end_month   = req.end_month,
            new_discount= req.new_discount
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Main chatbot endpoint.
    Returns text response, optional what-if result for chart, and route.
    """
    try:
        text, result, route = chat(
            user_question = req.message,
            collection    = collection,
            client        = llm_client
        )
        return {
            "text":   text,
            "result": result,
            "route":  route
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))