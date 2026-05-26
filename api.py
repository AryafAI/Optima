# api.py - FastAPI endpoints for the Optima DSS

import os
import traceback
import pandas as pd
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
    save_documents,
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

# Serve the interface folder as static files (only if it exists — the React
# frontend is now a separate project, so this is optional).
import os
if os.path.isdir("interface"):
    app.mount("/interface", StaticFiles(directory="interface"), name="interface")

# Load data, model, LLM client, and ChromaDB on startup
train_data, model, product_avg_price, pr_min, pr_max = load_all()
llm_client = create_llm_client(OPENAI_API_KEY)

# Validate that every configured PRODUCT_ID actually has rows for FIXED_STORE_ID
# in the training data. If a product is missing, what-if scenarios will fail at
# runtime with "No data found"; surfacing this at startup makes the issue easy
# to diagnose and fix in config.py.
print("\nValidating PRODUCT_IDS against training data...")
for _pid in PRODUCT_IDS:
    _matching = train_data[
        (train_data['Store ID']   == FIXED_STORE_ID) &
        (train_data['Product ID'] == _pid)
    ]
    _name = PRODUCT_NAMES.get(_pid, "Unknown")
    if len(_matching) == 0:
        print(f"  ⚠ WARNING: Product {_pid} ({_name}) has NO rows in Store "
              f"{FIXED_STORE_ID}. What-if scenarios for this product will fail. "
              f"Edit PRODUCT_IDS in config.py to use a product that exists in "
              f"this store.")
    else:
        _months = sorted(_matching['Month'].unique().tolist())
        _years  = sorted(_matching['Year'].unique().tolist())
        print(f"  Product {_pid} ({_name}): {len(_matching)} rows in Store "
              f"{FIXED_STORE_ID}, months {_months}, years {_years}")
print()

# Load or generate RAG documents.
# Validate that loaded documents reference the CURRENT PRODUCT_IDS - otherwise
# the RAG retrieval would return stale or empty results for current queries.
def _docs_match_current_products(metadatas):
    if not metadatas:
        return False
    seen_pids = {
        m.get('product_id')
        for m in metadatas
        if isinstance(m, dict) and m.get('product_id') is not None
    }
    # Require at least one document per current product
    return all(pid in seen_pids for pid in PRODUCT_IDS)

documents, metadatas, ids = [], [], []
try:
    documents, metadatas, ids = load_documents()
    if _docs_match_current_products(metadatas):
        print(f"Loaded {len(documents)} documents from backup (matches current products)")
    else:
        print("⚠ Loaded documents do not cover current PRODUCT_IDS - regenerating...")
        documents, metadatas, ids = [], [], []
        raise FileNotFoundError
except FileNotFoundError:
    print("No usable backup found - generating documents (this calls the LLM, may take ~30s)...")
    documents, metadatas, ids = generate_documents_with_llm(
        train_data   = train_data,
        store_id     = FIXED_STORE_ID,
        product_ids  = PRODUCT_IDS,
        call_llm     = lambda p: call_llm(llm_client, p)
    )
    try:
        save_documents(documents, metadatas, ids)
    except Exception as e:
        print(f"⚠ Could not save documents backup: {e}")

collection = setup_chromadb(documents, metadatas, ids)
print(f"ChromaDB ready with {collection.count()} documents")

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
    """
    Returns list of available products with their current selling price
    (from the most recent week of training data for each product).
    """
    out = []
    for pid in PRODUCT_IDS:
        try:
            baseline, _ = get_latest_baseline(FIXED_STORE_ID, pid)
            current_price = round(float(baseline.get('Avg_Price', 0.0)), 2)
        except Exception:
            current_price = None
        out.append({
            "id":            int(pid),
            "name":          PRODUCT_NAMES[pid],
            "current_price": current_price,
        })
    return out


@app.get("/overview")
def get_overview():
    """
    Returns the dashboard KPI overview computed from the latest 4 weeks
    of training data for the selected products at the fixed store:
        - total_quantity_sold
        - total_sales
        - total_profit  (sales - production cost*quantity, with a fallback)
        - top_product   (the product with the highest sales over those 4 weeks)
    """
    try:
        df = train_data[
            (train_data['Store ID']   == FIXED_STORE_ID) &
            (train_data['Product ID'].isin(PRODUCT_IDS))
        ].copy()

        if 'Week_Start' in df.columns:
            df['Week_Start'] = pd.to_datetime(df['Week_Start'], errors='coerce')

        # Keep latest 4 rows per product (most recent weeks)
        sort_col = 'Week_Start' if 'Week_Start' in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
        latest = df.groupby('Product ID', as_index=False).head(4)

        total_sales    = float(latest['Weekly_Sales'].sum())

        qty_col = next(
            (c for c in ['Quantity', 'Weekly_Quantity', 'Units_Sold', 'Lag1_Quantity']
             if c in latest.columns),
            None
        )
        if qty_col is not None:
            total_quantity = int(latest[qty_col].sum())
        else:
            # Fall back: derive units from sales / avg_price
            with_price = latest[latest['Avg_Price'] > 0]
            total_quantity = int((with_price['Weekly_Sales'] / with_price['Avg_Price']).sum())

        # Profit = sales - production cost × quantity, where Production Cost is per unit
        prod_cost_col = 'Production Cost' if 'Production Cost' in latest.columns else None
        if prod_cost_col and qty_col:
            total_cost   = float((latest[prod_cost_col] * latest[qty_col]).sum())
            total_profit = total_sales - total_cost
        elif prod_cost_col:
            # Approximate quantity from price when no quantity column
            with_price = latest[latest['Avg_Price'] > 0].copy()
            with_price['_qty'] = with_price['Weekly_Sales'] / with_price['Avg_Price']
            total_cost   = float((with_price[prod_cost_col] * with_price['_qty']).sum())
            total_profit = total_sales - total_cost
        else:
            total_profit = total_sales  

        # Top product by total sales over the latest 4 weeks
        per_product = latest.groupby('Product ID')['Weekly_Sales'].sum().sort_values(ascending=False)
        top_pid     = int(per_product.index[0])
        top_sales   = float(per_product.iloc[0])

        return {
            "total_quantity_sold": total_quantity,
            "total_sales":         round(total_sales,  2),
            "total_profit":        round(total_profit, 2),
            "top_product": {
                "id":    top_pid,
                "name":  PRODUCT_NAMES.get(top_pid, f"Product {top_pid}"),
                "sales": round(top_sales, 2),
            },
            "weeks_used": int(latest.groupby('Product ID').size().max()),
            "products":   [int(p) for p in PRODUCT_IDS],
        }
    except Exception as e:
        _log_and_raise(e)


def _log_and_raise(e, code=500):
    """Print the full traceback to the uvicorn console so we can debug,
    and surface the error type + message in the HTTP response."""
    tb = traceback.format_exc()
    print("\n" + "=" * 70, flush=True)
    print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
    print(tb, flush=True)
    print("=" * 70 + "\n", flush=True)
    raise HTTPException(status_code=code, detail=f"{type(e).__name__}: {e}")


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
        _log_and_raise(e)


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
        _log_and_raise(e)


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
        _log_and_raise(e)


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
        _log_and_raise(e)