# chatbot.py - Routing, RAG answering, and what-if chat handling for Optima

import json
import re
from openai import OpenAI
from config import OPENAI_MODEL, FIXED_STORE_ID, PRODUCT_NAMES
from whatif import (
    get_latest_baseline,
    whatif_price_change,
    whatif_avg_discount,
    whatif_extended_discount
)
from knowledge_base import retrieve_statistics_context


def _safe_json_loads(text):
    """Parse JSON from an LLM response, tolerating ```json ... ``` markdown fences,
    leading/trailing prose, and other formatting quirks. Returns None on failure."""
    if not text:
        return None
    s = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
    # Fall back to extracting the first {...} blob if there's surrounding prose
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# LLM Setup

def create_llm_client(api_key):
    """Creates and returns an OpenAI client."""
    return OpenAI(api_key=api_key)

def call_llm(client, prompt):
    """Calls the LLM and returns the response text."""
    response = client.chat.completions.create(
        model       = OPENAI_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0
    )
    return response.choices[0].message.content.strip()


# Query Router
def classify_user_query(user_question, client):
    """
    Routes user question to one of three paths:
    - what_if: user wants to simulate a scenario
    - statistics_rag: user asks about business performance
    - unknown: unclear or unrelated question
    """
    prompt = f"""
You are the routing layer for Optima, a retail decision support chatbot.

Classify the user question into exactly ONE route:

1. "what_if"
Use when the user wants to simulate a decision or test a scenario.
Examples:
- What if we increase the price of the Wedding Dress by 5 SAR?
- Apply 25% discount to the Graduation Dress
- What happens if I run 45% discount on Party Dress for 3 months?

2. "statistics_rag"
Use when the user asks about business performance, sales, revenue,
best/worst months, discount history, or product comparisons.

3. "unknown"
Use when the question is unclear, unrelated to retail, or just a greeting.

Rules:
- Return ONLY valid JSON
- Do not explain
- JSON must contain: route, reason

User question:
{user_question}

Expected JSON:
{{
  "route": "what_if",
  "reason": "The user is asking to simulate a scenario."
}}
"""
    response_text = call_llm(client, prompt)
    parsed = _safe_json_loads(response_text)
    if parsed is None:
        print(f"[router] failed to parse LLM response: {response_text!r}", flush=True)
        return {"route": "unknown", "reason": "Failed to parse routing response."}
    return parsed


# What-If Handler

def parse_whatif_parameters(user_question, client):
    """
    Extracts what-if parameters from the user question using LLM.
    Supports: price_change, discount_change, extended_discount
    """
    prompt = f"""
You are extracting structured what-if scenario parameters
for a retail decision support system.

Supported scenarios:
1. price_change — user wants to increase price by a SAR amount
2. discount_change — user wants to apply a discount rate
3. extended_discount — user wants to apply a discount over multiple months

Rules:
- Return ONLY valid JSON
- No explanation
- Discount must be decimal format (25% -> 0.25)
- Only valid discount values: 0, 0.25, 0.35, 0.45
- price_change value is always a positive SAR amount to ADD to current price
- For extended_discount include start_month and end_month as integers (1-12)
- MUST include: scenario, product_id, value
- If missing info, return empty JSON: {{}}

Product name to ID mapping:
{json.dumps({v: k for k, v in PRODUCT_NAMES.items()}, indent=2)}

Expected JSON examples:

Price change:
{{
  "scenario": "price_change",
  "product_id": 8999,
  "value": 10
}}

Discount change:
{{
  "scenario": "discount_change",
  "product_id": 12717,
  "value": 0.25
}}

Extended discount:
{{
  "scenario": "extended_discount",
  "product_id": 10013,
  "value": 0.45,
  "start_month": 3,
  "end_month": 5
}}

User question:
{user_question}
"""
    response_text = call_llm(client, prompt)
    data = _safe_json_loads(response_text)
    if data is None:
        print(f"[whatif-parser] failed to parse LLM response: {response_text!r}", flush=True)
        return None
    if not data or 'scenario' not in data:
        return None
    if 'product_id' not in data or 'value' not in data:
        return None
    return data


def generate_whatif_llm_response(result, client):
    """
    Generates a user-friendly business response from what-if result dict.
    """
    # Re-shape the result so every numeric field is unambiguously labeled with its
    # unit. The chart already renders the raw numbers — the LLM just narrates them.
    annotated = _annotate_units(result)

    prompt = f"""
You are Optima's business assistant for a fashion retail chain. Write a clear
business answer about the simulation result below in EXACTLY THREE PARAGRAPHS,
separated by blank lines.

REQUIRED FORMAT (follow this exactly — three paragraphs, blank line between each):

Paragraph 1 — describe what the scenario is, naming the product and the
specific change. Mention the actual SAR values involved (old and new price, or
old and new discount). Mention whether this is expected to grow or shrink sales.
1–2 sentences.

[blank line]

Paragraph 2 — give the sales numbers. Write something like:
"We're now expecting sales to rise from <baseline> SAR to <predicted> SAR,
reflecting an increase of <pct>%."  (or "drop" / "decrease" if pct is negative).
1 sentence.

[blank line]

Paragraph 3 — a short business conclusion. What does this mean for the store?
Examples: "customers are still willing to purchase the dress despite the higher
price", or "the discount may be too aggressive given the predicted drop". 1–2 sentences.

STRICT RULES:
- Output the three paragraphs ONLY, separated by blank lines.
- Do NOT output bullet points. Do NOT output headers. Do NOT output JSON.
- Do NOT echo the data dict back. Translate it into prose.
- Refer to the product by its name (e.g., "the Wedding Dress"), never by ID.

UNIT RULES (very important):
- Every monetary field ends in "_sar" and represents SAR (Saudi riyals), NEVER units.
- The dataset has NO unit/quantity column — never say "units", "items sold",
  "pieces" or similar. Every sales value MUST be written as "SAR <amount>".
- Numbers ending in "_pct" are percentages.

Be honest: if difference_pct or delta_pct is negative, the entire tone of all three
paragraphs should reflect a predicted DROP in sales (not growth).
Do NOT mention "model", "prediction", "code", or any technical terms.

Simulation data (translate this into the three-paragraph format above):
{json.dumps(annotated, indent=2)}

Now write the answer:
"""
    text = call_llm(client, prompt)
    # Defensive layer: if the LLM echoes JSON, uses wrong units, or fails to
    # produce three paragraphs, fall back to the deterministic summarizer so
    # the user always sees the same clean shape.
    s = (text or '').strip()
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
    if s.startswith('{') and s.endswith('}'):
        return _summarize_whatif_result(result)
    if _looks_like_unit_confusion(s):
        print(f"[whatif-llm] LLM mentioned units — falling back to deterministic summary. raw={s!r}",
              flush=True)
        return _summarize_whatif_result(result)
    # Strip stray bullets if the LLM added them despite the prompt — we want
    # plain prose paragraphs only.
    s = re.sub(r'^\s*[•·\-\*]\s*', '', s, flags=re.MULTILINE)
    # Require at least 3 paragraphs (i.e., 2 blank-line separators). If the LLM
    # returned a wall of text or only two paragraphs, fall back to the
    # deterministic three-paragraph version.
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', s) if p.strip()]
    if len(paragraphs) < 3:
        print(f"[whatif-llm] LLM returned {len(paragraphs)} paragraph(s); using structured fallback.",
              flush=True)
        return _summarize_whatif_result(result)
    return '\n\n'.join(paragraphs[:3])


def _annotate_units(result):
    """Return a copy of the result dict with `_sar`/`_pct` suffixes on numeric
    fields so the LLM cannot confuse currency for unit counts."""
    if not isinstance(result, dict):
        return result
    rename_map = {
        'old_price':       'old_price_sar',
        'new_price':       'new_price_sar',
        'price_increase':  'price_increase_sar',
        'baseline_sales':  'baseline_sales_sar',
        'new_sales':       'new_sales_sar',
        'difference':      'difference_sar',
    }
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _annotate_units(v)
        elif isinstance(v, list):
            out[k] = [_annotate_units(x) if isinstance(x, dict) else x for x in v]
        else:
            out[rename_map.get(k, k)] = v
    return out


def _looks_like_unit_confusion(text):
    """Detect when the LLM described monetary sales as discrete units."""
    if not text:
        return False
    t = text.lower()
    # A sentence is suspect if it pairs a sales-related noun with "units" / "items" /
    # "pieces" — without "SAR" anywhere nearby.
    bad_terms = ['units', 'items sold', 'pieces sold', 'pcs', 'unit sales']
    if any(term in t for term in bad_terms) and 'sar' not in t:
        return True
    return False


def _summarize_whatif_result(result):
    """Code-side three-paragraph fallback. Used when the LLM misbehaves or
    fails to produce the right shape. Mirrors the prompt template in
    generate_whatif_llm_response above. Edit the f-strings below to change
    the deterministic answer text."""
    name     = result.get('product_name', 'the product')
    scenario = result.get('scenario')

    def _verb(pct, growing, dropping):
        return growing if pct >= 0 else dropping

    def _conclusion_price(pct, name):
        if pct >= 3:
            return (f"This change suggests that customers are still willing to "
                    f"purchase the {name} despite the higher price.")
        if pct > 0:
            return (f"The {name} appears mostly resilient to the higher price, "
                    f"but the upside is small.")
        if pct >= -3:
            return (f"The {name} loses a small amount of weekly sales at the "
                    f"new price — worth monitoring before committing.")
        return (f"Customers appear sensitive to the higher price for the {name}, "
                f"so this increase is risky and should be reviewed.")

    def _conclusion_discount(pct, name):
        if pct >= 3:
            return (f"The deeper discount appears to drive enough additional "
                    f"weekly sales of the {name} to be worthwhile.")
        if pct > 0:
            return (f"The discount lifts {name} sales only slightly — its "
                    f"margin impact may not justify the change.")
        if pct >= -3:
            return (f"The discount produces little uplift for the {name}; the "
                    f"price reduction is likely cutting into margin without "
                    f"meaningful volume growth.")
        return (f"The discount is predicted to actually shrink revenue for the "
                f"{name}; consider keeping the current pricing.")

    def _conclusion_extended(pct, name):
        if pct >= 3:
            return (f"Running this campaign across the selected period looks "
                    f"effective for the {name} and should grow total sales.")
        if pct > 0:
            return (f"The campaign produces a small uplift for the {name} — "
                    f"weigh it against the lost margin before launching.")
        if pct >= -3:
            return (f"The campaign barely moves total sales for the {name}; "
                    f"the discount may not be worth the margin sacrifice.")
        return (f"This extended discount is predicted to reduce total revenue "
                f"for the {name}; review the campaign before launching.")

    if scenario == 'price_change':
        old_p     = result.get('old_price', 0)
        new_p     = result.get('new_price', 0)
        b         = result.get('baseline_sales', 0)
        n         = result.get('new_sales', 0)
        pct       = result.get('difference_pct', 0)
        verb_para1 = _verb(pct, "a boost", "a drop")
        verb_para2 = _verb(pct, "rise",   "drop")
        verb_pct   = _verb(pct, "increase", "decrease")
        sign_pct   = abs(pct)
        para1 = (f"The {name} is seeing a price increase from {old_p:.2f} SAR "
                 f"to {new_p:.2f} SAR, which has led to {verb_para1} in weekly sales.")
        para2 = (f"We're now expecting sales to {verb_para2} from {b:.2f} SAR "
                 f"to {n:.2f} SAR, reflecting a{('n' if verb_pct[0] in 'aeiou' else '')} "
                 f"{verb_pct} of {sign_pct:.1f}%.")
        para3 = _conclusion_price(pct, name)
        return f"{para1}\n\n{para2}\n\n{para3}"

    if scenario == 'discount_change':
        old_d     = result.get('old_discount_pct', 0)
        new_d     = result.get('new_discount_pct', 0)
        b         = result.get('baseline_sales', 0)
        n         = result.get('new_sales', 0)
        pct       = result.get('difference_pct', 0)
        verb_para2 = _verb(pct, "rise", "drop")
        verb_pct   = _verb(pct, "increase", "decrease")
        sign_pct   = abs(pct)
        para1 = (f"The discount on the {name} is changing from {old_d:.0f}% "
                 f"to {new_d:.0f}%, which is expected to "
                 f"{_verb(pct, 'lift', 'reduce')} weekly sales.")
        para2 = (f"We're now expecting sales to {verb_para2} from {b:.2f} SAR "
                 f"to {n:.2f} SAR, reflecting a{('n' if verb_pct[0] in 'aeiou' else '')} "
                 f"{verb_pct} of {sign_pct:.1f}%.")
        para3 = _conclusion_discount(pct, name)
        return f"{para1}\n\n{para2}\n\n{para3}"

    if scenario == 'extended_discount':
        d         = result.get('discount_pct', 0)
        sm        = result.get('start_month', '')
        em        = result.get('end_month', '')
        total     = result.get('total', {}) or {}
        b         = total.get('baseline_sales', 0)
        n         = total.get('new_sales', 0)
        pct       = total.get('delta_pct', 0)
        verb_para2 = _verb(pct, "rise", "drop")
        verb_pct   = _verb(pct, "increase", "decrease")
        sign_pct   = abs(pct)
        para1 = (f"Running a {d:.0f}% discount on the {name} from {sm} to "
                 f"{em} is expected to {_verb(pct, 'lift', 'reduce')} total sales.")
        para2 = (f"We're now expecting total sales to {verb_para2} from "
                 f"{b:.2f} SAR to {n:.2f} SAR over the period, reflecting "
                 f"a{('n' if verb_pct[0] in 'aeiou' else '')} {verb_pct} of "
                 f"{sign_pct:.1f}%.")
        para3 = _conclusion_extended(pct, name)
        return f"{para1}\n\n{para2}\n\n{para3}"

    return "Simulation completed. See the chart for the predicted impact."


def handle_whatif_question(user_question, client):
    """
    Full what-if execution flow.
    Returns (text_response, result_dict) so the interface can render the chart.
    result_dict is None for unknown or failed scenarios.
    """
    params = parse_whatif_parameters(user_question, client)

    if params is None:
        return "I could not understand the request. Please specify the product and what you want to change.", None

    scenario   = params.get('scenario')
    product_id = params.get('product_id')
    value      = params.get('value')

    if product_id is None or value is None:
        return "Missing required information. Please include the product and the value.", None

    try:
        if scenario == 'price_change':
            baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, product_id)
            result = whatif_price_change(baseline, baseline_pred, value)

        elif scenario == 'discount_change':
            baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, product_id)
            result = whatif_avg_discount(baseline, baseline_pred, value)

        elif scenario == 'extended_discount':
            start_month = params.get('start_month')
            end_month   = params.get('end_month')
            if start_month is None or end_month is None:
                return "Please specify the start and end month for the extended discount.", None
            result = whatif_extended_discount(
                store_id    = FIXED_STORE_ID,
                product_id  = product_id,
                start_month = start_month,
                end_month   = end_month,
                new_discount= value
            )
            if result.get('status') == 'no_data':
                return result['message'], None

        else:
            return "This scenario is not supported.", None

        text = generate_whatif_llm_response(result, client)
        return text, result

    except ValueError as e:
        return f"Invalid input: {e}", None
    except Exception as e:
        return f"Something went wrong: {e}", None


# Statistics RAG Handler
def answer_statistics_rag_question(user_question, collection, client):
    """
    Answers statistics questions using retrieved RAG context.
    """
    context = retrieve_statistics_context(user_question, collection)

    # The retrieve function already falls back through looser filters; an empty
    # result here means the ChromaDB collection itself is empty.
    if not context or not context.strip():
        return ("I couldn't find any business documents in the knowledge base. "
                "Try restarting the backend so the documents are regenerated, "
                "or check that the data files are reachable.")

    prompt = f"""
You are Optima's business analytics chatbot.

Answer the user's question using ONLY the retrieved business context.

Rules:
- Do not invent numbers
- Do not use information outside the context
- Do not mention technical details, embeddings, or code
- If the context is not enough, say the data is not sufficient
- Keep the answer clear, professional, and business-oriented
- Use product names not product IDs

User question:
{user_question}

Retrieved context:
{context}
"""
    return call_llm(client, prompt)


# Main Chat Function
def chat(user_question, collection, client):
    """
    Main chatbot entry point.
    Returns (text_response, result_dict, route)
    - text_response: message to display in chat
    - result_dict: what-if result for chart rendering (None for stats questions)
    - route: 'what_if', 'statistics_rag', or 'unknown'
    """
    route_info = classify_user_query(user_question, client)
    route      = route_info.get('route', 'unknown')

    if route == 'what_if':
        text, result = handle_whatif_question(user_question, client)
        return text, result, route

    elif route == 'statistics_rag':
        text = answer_statistics_rag_question(user_question, collection, client)
        return text, None, route

    return "Sorry, I could not understand your request.", None, route