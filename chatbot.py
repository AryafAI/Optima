# chatbot.py - Routing, RAG answering, and what-if chat handling for Optima

import json
from openai import OpenAI
from config import OPENAI_MODEL, FIXED_STORE_ID, PRODUCT_NAMES
from whatif import (
    get_latest_baseline,
    whatif_price_change,
    whatif_avg_discount,
    whatif_extended_discount
)
from knowledge_base import retrieve_statistics_context

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
    try:
        return json.loads(response_text)
    except Exception:
        return {"route": "unknown", "reason": "Failed to parse routing response."}


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
    try:
        data = json.loads(response_text)
        if not data or 'scenario' not in data:
            return None
        if 'product_id' not in data or 'value' not in data:
            return None
        return data
    except Exception:
        return None


def generate_whatif_llm_response(result, client):
    """
    Generates a user-friendly business response from what-if result dict.
    """
    prompt = f"""
You are Optima's business assistant for a fashion retail chain.

Rules:
- Use simple business language
- Keep it short and clear
- Do NOT mention models, code, or technical details
- Use ONLY the provided numbers
- Use the product name not the product ID
- Mention scenario type and key numbers

Result:
{json.dumps(result, indent=2)}

Final Answer:
"""
    return call_llm(client, prompt)


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

    if not context or len(context.strip()) < 50:
        return "Sorry, the available data is not sufficient to answer this question."

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