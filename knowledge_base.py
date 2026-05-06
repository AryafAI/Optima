# knowledge_base.py - Document generation, ChromaDB setup, and RAG retrieval

import re
import json
import chromadb
import chromadb.utils
from config import (
    PRODUCT_NAMES, MONTH_NAMES, FIXED_STORE_ID, PRODUCT_IDS,
    RAG_DOCS_FILE, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL
)


# Product Name Helper
def get_product_name(product_id):
    """Returns product name from PRODUCT_NAMES dict. Falls back to ID."""
    return PRODUCT_NAMES.get(product_id, f"Product {product_id}")


# Document Generation

# 1. Business text Generation
def generate_business_text_with_llm(doc_type, facts, call_llm):
    """
    Converts trusted facts into a professional business paragraph using LLM.
    call_llm is passed as argument to avoid circular imports with chatbot.py.
    """
    prompt = f"""
You are a senior retail business analyst writing business knowledge documents
for Optima Decision Support System.

Your task:
Write ONE professional business paragraph using ONLY the provided facts.

Strict rules:
- Do NOT invent numbers
- Do NOT change or rename the product name — use it EXACTLY as provided
- Do NOT add unsupported assumptions
- Do NOT mention AI, machine learning, code, variables, or technical details
- Use professional business language
- Keep it concise and clear
- Focus only on the requested document type

Document Type:
{doc_type}

Trusted Facts:
{json.dumps(facts, indent=2, ensure_ascii=False)}

Final Business Document:
"""
    return call_llm(prompt)


# 2. Document Generation
def generate_documents_with_llm(train_data, store_id, product_ids, call_llm):
    """
    Generates 5 document types per product set:
    1. Product Summary
    2. Best/Worst Month
    3. Monthly Breakdown
    4. Product Comparison
    5. Store Monthly Insights
    """
    documents = []
    metadatas = []
    ids       = []

    store_data = train_data[
        (train_data['Store ID']   == store_id) &
        (train_data['Product ID'].isin(product_ids))
    ].copy()

    if len(store_data) == 0:
        print("No matching data found.")
        return documents, metadatas, ids

    for product_id in product_ids:
        subset = store_data[store_data['Product ID'] == product_id].copy()

        if len(subset) == 0:
            print(f"Skipping {get_product_name(product_id)} — no data")
            continue

        product_name     = get_product_name(product_id)
        total_revenue    = float(subset['Weekly_Sales'].sum())
        avg_price        = float(subset['Avg_Price'].mean())
        avg_discount     = float(subset['Avg_Discount'].mean() * 100)
        max_campaign     = float(subset['Campaign_Discount'].max() * 100)
        months_available = sorted(subset['Month'].dropna().unique().tolist())
        monthly_revenue  = (
            subset.groupby('Month')['Weekly_Sales']
            .sum()
            .sort_values(ascending=False)
        )
        best_month  = int(monthly_revenue.index[0])
        worst_month = int(monthly_revenue.index[-1])

        # 1. Product Summary
        facts_summary = {
            'product_name':                       product_name,
            'product_id':                         int(product_id),
            'store_id':                           int(store_id),
            'total_revenue_sar':                  round(total_revenue, 2),
            'average_price_sar':                  round(avg_price, 2),
            'average_discount_percent':           round(avg_discount, 2),
            'maximum_campaign_discount_percent':  round(max_campaign, 2),
            'available_months':                   [MONTH_NAMES[m] for m in months_available],
        }
        documents.append(generate_business_text_with_llm('product_summary', facts_summary, call_llm))
        metadatas.append({'type': 'product_summary',  'store_id': int(store_id), 'product_id': int(product_id)})
        ids.append(f"summary_{store_id}_{product_id}")

        # 2. Best/Worst Month
        facts_best_worst = {
            'product_name':            product_name,
            'product_id':              int(product_id),
            'best_month':              MONTH_NAMES[best_month],
            'best_month_revenue_sar':  round(float(monthly_revenue.loc[best_month]), 2),
            'worst_month':             MONTH_NAMES[worst_month],
            'worst_month_revenue_sar': round(float(monthly_revenue.loc[worst_month]), 2),
        }
        documents.append(generate_business_text_with_llm('best_worst_month', facts_best_worst, call_llm))
        metadatas.append({'type': 'best_worst_month', 'store_id': int(store_id), 'product_id': int(product_id)})
        ids.append(f"bestworst_{store_id}_{product_id}")

        # 3. Monthly Breakdown
        monthly_breakdown = {
            MONTH_NAMES[int(month)]: round(float(revenue), 2)
            for month, revenue in monthly_revenue.items()
        }
        facts_monthly = {
            'product_name':       product_name,
            'product_id':         int(product_id),
            'store_id':           int(store_id),
            'monthly_revenue_sar': monthly_breakdown,
        }
        documents.append(generate_business_text_with_llm('monthly_breakdown', facts_monthly, call_llm))
        metadatas.append({'type': 'monthly_breakdown', 'store_id': int(store_id), 'product_id': int(product_id)})
        ids.append(f"monthly_{store_id}_{product_id}")

    # 4. Product Comparison
    revenue_rank      = store_data.groupby('Product ID')['Weekly_Sales'].sum().sort_values(ascending=False)
    top_product_id    = int(revenue_rank.index[0])
    bottom_product_id = int(revenue_rank.index[-1])

    facts_comparison = {
        'store_id': int(store_id),
        'top_product': {
            'product_id':   top_product_id,
            'product_name': get_product_name(top_product_id),
            'revenue_sar':  round(float(revenue_rank.iloc[0]), 2),
        },
        'lowest_product': {
            'product_id':   bottom_product_id,
            'product_name': get_product_name(bottom_product_id),
            'revenue_sar':  round(float(revenue_rank.iloc[-1]), 2),
        },
    }
    documents.append(generate_business_text_with_llm('product_comparison', facts_comparison, call_llm))
    metadatas.append({'type': 'product_comparison', 'store_id': int(store_id)})
    ids.append(f"comparison_{store_id}")

    # 5. Store Monthly Insights
    store_monthly_revenue = (
        store_data.groupby('Month')['Weekly_Sales']
        .sum()
        .sort_values(ascending=False)
    )
    store_best_month  = int(store_monthly_revenue.index[0])
    store_worst_month = int(store_monthly_revenue.index[-1])

    facts_store = {
        'store_id':               int(store_id),
        'best_month':             MONTH_NAMES[store_best_month],
        'best_month_revenue_sar': round(float(store_monthly_revenue.loc[store_best_month]), 2),
        'worst_month':            MONTH_NAMES[store_worst_month],
        'worst_month_revenue_sar':round(float(store_monthly_revenue.loc[store_worst_month]), 2),
        'monthly_revenue_sar':    {
            MONTH_NAMES[int(m)]: round(float(r), 2)
            for m, r in store_monthly_revenue.items()
        },
    }
    documents.append(generate_business_text_with_llm('store_monthly_insights', facts_store, call_llm))
    metadatas.append({'type': 'store_monthly_insights', 'store_id': int(store_id)})
    ids.append(f"store_monthly_{store_id}")

    print(f"Generated {len(documents)} documents successfully.")
    return documents, metadatas, ids


def save_documents(documents, metadatas, ids):
    """Saves generated documents to Drive as a JSON backup."""
    with open(RAG_DOCS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'documents': documents, 'metadatas': metadatas, 'ids': ids},
                  f, ensure_ascii=False, indent=2)
    print(f"Documents saved to {RAG_DOCS_FILE}")


def load_documents():
    """Loads previously saved documents from Drive."""
    with open(RAG_DOCS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data['documents'])} documents from backup")
    return data['documents'], data['metadatas'], data['ids']


# ChromaDB Setup
def setup_chromadb(documents, metadatas, ids):
    """
    Embeds documents and stores them in ChromaDB.
    Clears existing collection before adding new documents.
    Returns the collection.
    """
    embedding_fn  = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    chroma_client = chromadb.Client()
    collection    = chroma_client.get_or_create_collection(
        name               = CHROMA_COLLECTION_NAME,
        embedding_function = embedding_fn
    )

    existing = collection.get()
    if len(existing['ids']) > 0:
        collection.delete(ids=existing['ids'])

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"ChromaDB ready - {collection.count()} documents")
    return collection


# Retrieval
def extract_product_id_from_question(question):
    """Extracts product ID from question — supports both numeric ID and name."""
    match = re.search(r'product\s+(\d+)', question.lower())
    if match:
        return int(match.group(1))
    for pid, name in PRODUCT_NAMES.items():
        if name.lower() in question.lower():
            return pid
    return None


def detect_query_type(question):
    """Detects the most relevant document type for this question."""
    q = question.lower().strip()

    # Top/best-selling product comparisons must be checked BEFORE the generic
    # product_summary keywords below, so questions like "what is the best
    # selling product?" route to the dedicated product_comparison document
    # (which has top vs bottom by revenue) instead of returning a noisy
    # mix of unrelated summaries.
    if any(p in q for p in [
        'best selling', 'best-selling', 'top selling', 'top-selling',
        'highest selling', 'most sold', 'most popular', 'top performer',
        'top performing product',
    ]):
        return 'product_comparison'

    if any(w in q for w in ['compare', 'comparison', 'versus',
                             'vs', 'higher than', 'lower than',
                             'rank', 'ranking', 'which product is']):
        return 'product_comparison'

    if 'total revenue' in q:
        return 'product_summary'

    if any(w in q for w in ['how much', 'overall', 'total sales', 'generated',
                             'discount history', 'has discount', 'campaign history',
                             'highest revenue', 'most revenue',
                             'which product', 'best product']):
        return 'product_summary'

    if any(w in q for w in ['best month', 'worst month', 'peak',
                             'highest month', 'lowest month', 'performing month']):
        return 'best_worst_month'

    if any(w in q for w in ['in january', 'in february', 'in march', 'in april',
                             'in may', 'in june', 'in july', 'in august',
                             'in september', 'in october', 'in november',
                             'in december', 'sales in', 'revenue in']):
        return 'monthly_breakdown'

    if any(w in q for w in ['overall store', 'store performance', 'store revenue']):
        return 'store_monthly_insights'

    return None


def retrieve_statistics_context(user_question, collection, n_results=3):
    """
    Retrieves the most relevant business documents from ChromaDB.
    Tries a strict (product+type) filter first, then progressively looser ones,
    so we never return empty unless the collection itself is empty.
    """
    product_id = extract_product_id_from_question(user_question)
    doc_type   = detect_query_type(user_question)
    total      = collection.count()
    n          = min(n_results, total) if total else 0

    print(f"[rag] question={user_question!r} product_id={product_id} "
          f"doc_type={doc_type} collection_count={total}", flush=True)

    if total == 0:
        print("[rag] collection is empty — no documents to retrieve", flush=True)
        return ''

    # Try filters from strictest to loosest. Stop at the first one that returns docs.
    filter_attempts = []
    if product_id and doc_type:
        filter_attempts.append({'$and': [{'product_id': product_id}, {'type': doc_type}]})
    if product_id:
        filter_attempts.append({'product_id': product_id})
    if doc_type:
        filter_attempts.append({'type': doc_type})
    filter_attempts.append(None)  # no filter — pure semantic search

    docs, metas = [], []
    for where in filter_attempts:
        try:
            results = collection.query(
                query_texts = [user_question],
                n_results   = n,
                where       = where
            )
            d = results['documents'][0]
            m = results['metadatas'][0]
            print(f"[rag] filter={where} → {len(d)} hits", flush=True)
            if d:
                docs, metas = d, m
                break
        except Exception as e:
            print(f"[rag] query failed for filter={where}: {type(e).__name__}: {e}", flush=True)

    if not docs:
        return ''

    context_blocks = [
        f"Metadata: {meta}\nContent:\n{doc}"
        for doc, meta in zip(docs, metas)
    ]
    context = '\n\n---\n\n'.join(context_blocks)
    print(f"[rag] returning {len(context)} chars of context", flush=True)
    return context