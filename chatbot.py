# chatbot.py - Routing, RAG answering, and what-if chat handling for Optima

import json
import re
import difflib
from openai import OpenAI
from config import OPENAI_MODEL, FIXED_STORE_ID, PRODUCT_NAMES, VALID_DISCOUNTS, MONTH_NAMES
from whatif import (
    get_latest_baseline,
    whatif_price_change,
    whatif_avg_discount,
    whatif_extended_discount
)
from knowledge_base import retrieve_statistics_context


def _safe_json_loads(text):
    """Parse JSON from an LLM response, Returns None on failure."""
    if not text:
        return None
    s = text.strip()
    if s.startswith('```'):
        s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
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
    Extracts what-if scenario parameters from the user question using the LLM.
    Always returns a dict with the five keys below (missing values are None) so
    the caller can give a SPECIFIC, helpful error. Returns None only when the
    LLM response is completely unreadable.
    Supports: price_change, discount_change, extended_discount.
    """
    prompt = f"""
You are extracting structured what-if scenario parameters
for a retail decision support system.

Supported scenarios:
1. price_change - the user wants to RAISE the selling price by a SAR amount.
2. discount_change - the user wants to apply ONE discount rate (no months).
3. extended_discount - the user wants a discount applied across a RANGE of
   months (two or more months, or a "for N months" period).

How to choose the scenario:
- If the user mentions a month range, two months, a season, a quarter, or any
  multi-month period in ANY wording, choose "extended_discount" - even if the
  word "extend" is NOT used. These are ALL extended_discount:
  "from March to May", "between March and May", "March through May",
  "for 3 months", "during spring", "over Q2", "apply 25% Jan to April".
- A discount with no month range is "discount_change".
- Raising the price is "price_change".

Extraction rules:
- Return ONLY valid JSON. No explanation, no markdown.
- ALWAYS return the JSON object shown below. If a field is missing or unclear,
  set it to null - do NOT drop the field and do NOT return an empty object.
- "product_id": map the product name to its ID using the mapping below; match
  even if the name is slightly misspelled. If you truly cannot tell, use null.
- "value": for price_change a positive SAR number to ADD to the price; for
  discount_change / extended_discount the discount as a decimal (25% -> 0.25).
  Return the number the user actually said even if it is unusual - do NOT round
  it, do NOT reject it, do NOT replace it. If absent, use null.
- "start_month" / "end_month": integers 1-12 for extended_discount, else null.

Product name to ID mapping:
{json.dumps({v: k for k, v in PRODUCT_NAMES.items()}, indent=2)}

Always return EXACTLY this shape:
{{
  "scenario": "price_change" | "discount_change" | "extended_discount" | null,
  "product_id": <int or null>,
  "value": <number or null>,
  "start_month": <int 1-12 or null>,
  "end_month": <int 1-12 or null>
}}

User question:
{user_question}
"""
    response_text = call_llm(client, prompt)
    data = _safe_json_loads(response_text)
    if not isinstance(data, dict):
        print(f"[whatif-parser] failed to parse LLM response: {response_text!r}", flush=True)
        return None
    # Normalise: guarantee all five keys exist so the caller can inspect each
    # one and produce a specific message about whatever is missing or wrong.
    return {
        'scenario':    data.get('scenario'),
        'product_id':  data.get('product_id'),
        'value':       data.get('value'),
        'start_month': data.get('start_month'),
        'end_month':   data.get('end_month'),
    }


def generate_whatif_llm_response(result, client):
    """
    Generates a user-friendly business response from what-if result dict.
    """
    # Re-shape the result so every numeric field is unambiguously labeled with its
    # unit. The chart already renders the raw numbers - the LLM just narrates them.
    annotated = _annotate_units(result)

    prompt = f"""
You are Optima's business assistant for a fashion retail chain. Write a clear
business answer about the simulation result below in EXACTLY THREE PARAGRAPHS,
separated by blank lines.

REQUIRED FORMAT (follow this exactly - three paragraphs, blank line between each):

Paragraph 1 - describe what is happening directly. Start with the product name
and the change itself. Mention the actual SAR values involved (old and new
price, or old and new discount). Mention whether this is expected to grow or
shrink sales. 1-2 sentences.

GOOD opening examples for paragraph 1:
- "The Wedding Dress is seeing a price increase from SAR 56.5 to SAR 66.5..."
- "The discount on the Party Dress is changing from 25% to 35%..."
- "Running a 45% discount on the Graduation Dress from March to May..."

BAD opening examples - DO NOT START LIKE THESE:
- "In this scenario, we are examining..."
- "This scenario explores..."
- "Let's analyze the impact of..."
- "We're looking at what happens when..."
- "This analysis shows..."

[blank line]

Paragraph 2 - give the sales numbers. Write something like:
"We're now expecting sales to rise from <baseline> SAR to <predicted> SAR,
reflecting an increase of <pct>%."  (or "drop" / "decrease" if pct is negative).
1 sentence.

[blank line]

Paragraph 3 - a short business conclusion. What does this mean for the store?
Examples: "customers are still willing to purchase the dress despite the higher
price", or "the discount may be too aggressive given the predicted drop". 1–2 sentences.

STRICT RULES:
- Output the three paragraphs ONLY, separated by blank lines.
- Do NOT output bullet points. Do NOT output headers. Do NOT output JSON.
- Do NOT echo the data dict back. Translate it into prose.
- Refer to the product by its name (e.g., "the Wedding Dress"), never by ID.
- Do NOT start with meta-phrases like "In this scenario", "This scenario",
  "Let's", "We are examining", "This analysis", "The simulation shows" - open
  paragraph 1 with the product name itself.

UNIT RULES (very important):
- Every monetary field ends in "_sar" and represents SAR (Saudi riyals), NEVER units.
- The dataset has NO unit/quantity column - never say "units", "items sold",
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
    # Strip stray bullets if the LLM added them despite the prompt - we want
    # plain prose paragraphs only.
    s = re.sub(r'^\s*[•·\-\*]\s*', '', s, flags=re.MULTILINE)
    # Require at least 3 paragraphs. If the LLM returned a wall of text or only 
    # two paragraphs, fall back to the deterministic three-paragraph version.
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', s) if p.strip()]
    if len(paragraphs) < 3:
        print(f"[whatif-llm] LLM returned {len(paragraphs)} paragraph(s); using structured fallback.",
              flush=True)
        return _summarize_whatif_result(result)
    # that opening sentence - keep the rest of the paragraph if there is more.
    paragraphs[0] = _strip_meta_opening(paragraphs[0])
    if not paragraphs[0]:
        paragraphs = paragraphs[1:]
        if len(paragraphs) < 3:
            return _summarize_whatif_result(result)
    return '\n\n'.join(paragraphs[:3])


_META_OPENING_PATTERNS = [
    r'^\s*in\s+this\s+scenario[^.]*\.\s*',
    r'^\s*this\s+scenario[^.]*\.\s*',
    r'^\s*let[\'s]+\s+(analyze|examine|look)[^.]*\.\s*',
    r'^\s*we[\'re]*\s+(examining|looking\s+at|analyzing)[^.]*\.\s*',
    r'^\s*this\s+analysis[^.]*\.\s*',
    r'^\s*the\s+simulation\s+shows[^.]*\.\s*',
    r'^\s*here[\'s]?\s+(an?\s+)?(analysis|breakdown)[^.]*\.\s*',
]

def _strip_meta_opening(paragraph):
    """Remove a leading meta-explanation sentence ("In this scenario, ...") if
    present. Leaves the rest of the paragraph intact."""
    out = paragraph
    for pat in _META_OPENING_PATTERNS:
        out = re.sub(pat, '', out, count=1, flags=re.IGNORECASE)
    return out.strip()


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
    bad_terms = ['units', 'items sold', 'pieces sold', 'pcs', 'unit sales']
    if any(term in t for term in bad_terms) and 'sar' not in t:
        return True
    return False


def _summarize_whatif_result(result):
    """Code-side three-paragraph fallback. Used when the LLM misbehaves or
    fails to produce the right shape. Mirrors the prompt template in
    generate_whatif_llm_response above."""
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



def _product_catalog_text():
    """Readable list of valid products, e.g. 'Wedding Dress, Graduation Dress, Party Dress'."""
    return ", ".join(PRODUCT_NAMES.values())


def _discount_catalog_text():
    """Readable list of valid discounts, e.g. '0%, 25%, 35%, 45%'."""
    return ", ".join(f"{int(round(d * 100))}%" for d in VALID_DISCOUNTS)


def _coerce_number(x):
    """Return x as a number. Accepts ints, floats, and numeric strings such as
    '0.25', '3', or '10 SAR'. Returns None when there is no number to read —
    so the caller can treat it as 'missing' instead of crashing."""
    if x is None or isinstance(x, (int, float)):
        return x
    m = re.search(r'-?\d+(?:\.\d+)?', str(x))
    if not m:
        return None
    s = m.group(0)
    return float(s) if '.' in s else int(s)


def _format_discount_value(value):
    """Show the discount the user asked for in a readable way for error text."""
    try:
        pct = value * 100 if value <= 1 else value
        return f"{pct:g}%"
    except Exception:
        return str(value)


def _suggest_product_name(user_question):
    """If the user seems to have mistyped a product name, return the closest
    valid product name. Returns None when nothing is close enough."""
    if not user_question:
        return None
    text = user_question.lower()
    valid_names = list(PRODUCT_NAMES.values())
    # 1) the full product name already appears in the text
    for name in valid_names:
        if name.lower() in text:
            return name
    # 2) fuzzy-match on the distinguishing word (Wedding / Graduation / Party)
    key_word = {name.split()[0].lower(): name for name in valid_names}
    for word in re.findall(r"[a-z]+", text):
        match = difflib.get_close_matches(word, list(key_word.keys()), n=1, cutoff=0.75)
        if match:
            return key_word[match[0]]
    return None


def handle_whatif_question(user_question, client):
    """
    Full what-if execution flow.
    Returns (text_response, result_dict). result_dict is None whenever the
    scenario could not be run.

    Each failure path returns a SPECIFIC, user-friendly message so the user
    knows exactly what to fix. Checks run from the most specific case to the
    most general; the final try/except is a safety net so the bot always
    replies with something readable instead of crashing.
    """
    params = parse_whatif_parameters(user_question, client)

    # The LLM reply could not be read at all - give a guiding fallback.
    if params is None:
        return (
            "I couldn't read that request. You can ask me to:\n"
            "1. Raise a price — e.g. \"Raise the Party Dress price by 10 SAR\".\n"
            "2. Change a discount — e.g. \"Apply 25% discount on the Wedding Dress\".\n"
            "3. Run a discount over months — e.g. \"Apply 25% discount on the "
            "Wedding Dress from March to May\".",
            None,
        )

    scenario    = params.get('scenario')
    product_id  = _coerce_number(params.get('product_id'))
    value       = _coerce_number(params.get('value'))
    start_month = _coerce_number(params.get('start_month'))
    end_month   = _coerce_number(params.get('end_month'))

    # 1) The scenario itself must be recognised.
    if scenario not in ('price_change', 'discount_change', 'extended_discount'):
        return (
            "I couldn't tell what you'd like to simulate. I can do three things:\n"
            "1. Raise a price (e.g. \"raise the Party Dress price by 10 SAR\").\n"
            "2. Change a discount (e.g. \"apply 25% discount on the Wedding Dress\").\n"
            "3. Run a discount across months (e.g. \"apply 25% discount on the "
            "Wedding Dress from March to May\").",
            None,
        )

    # 2) The product must exists.
    if product_id is None or product_id not in PRODUCT_NAMES:
        suggestion = _suggest_product_name(user_question)
        hint = f" Did you mean \"{suggestion}\"?" if suggestion else ""
        return (
            f"I couldn't find that product.{hint}\n"
            f"Please choose one of our products: {_product_catalog_text()}.",
            None,
        )

    product_id   = int(product_id)
    product_name = PRODUCT_NAMES[product_id]

    # 3) A value is required - a SAR amount, or a discount rate.
    if value is None:
        if scenario == 'price_change':
            return (
                f"How much would you like to raise the {product_name} price by? "
                "Please give an amount in SAR, e.g. \"by 10 SAR\".",
                None,
            )
        return (
            f"Which discount would you like to apply to the {product_name}?\n"
            f"Available discounts: {_discount_catalog_text()}.",
            None,
        )

    # 4) Scenario-specific validation - a message tailored to each problem.
    if scenario == 'price_change':
        if value <= 0:
            return (
                "The price increase has to be a positive amount in SAR, "
                "e.g. \"raise the price by 10 SAR\".",
                None,
            )
    else:
        # discount_change and extended_discount both need an allowed discount.
        if value not in VALID_DISCOUNTS:
            return (
                f"{_format_discount_value(value)} is not an available discount "
                f"for the {product_name}.\n"
                f"Please pick one of the allowed discounts: {_discount_catalog_text()}.",
                None,
            )

    if scenario == 'extended_discount':
        if start_month is None or end_month is None:
            return (
                f"Please tell me the period for the {product_name} discount — "
                "a start month and an end month, e.g. \"from March to May\".",
                None,
            )
        start_month = int(start_month)
        end_month   = int(end_month)
        if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
            return ("Please use months between 1 (January) and 12 (December).", None)
        if start_month > end_month:
            return (
                f"The start month ({MONTH_NAMES[start_month]}) comes after the "
                f"end month ({MONTH_NAMES[end_month]}). Please put the earlier "
                "month first, e.g. \"from March to May\".",
                None,
            )

    # 5) Everything checks out - run the simulation.
    try:
        if scenario == 'price_change':
            baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, product_id)
            result = whatif_price_change(baseline, baseline_pred, value)

        elif scenario == 'discount_change':
            baseline, baseline_pred = get_latest_baseline(FIXED_STORE_ID, product_id)
            result = whatif_avg_discount(baseline, baseline_pred, value)

        else:  # extended_discount
            result = whatif_extended_discount(
                store_id     = FIXED_STORE_ID,
                product_id   = product_id,
                start_month  = start_month,
                end_month    = end_month,
                new_discount = value,
            )
            if result.get('status') == 'no_data':
                return (
                    f"I don't have historical data for the {product_name} in the "
                    "months you picked, so I can't simulate that period. "
                    "Try a different month range.",
                    None,
                )

        text = generate_whatif_llm_response(result, client)
        return text, result

    # Safety net - anything not caught above still gets a readable reply
    # instead of crashing the chat.
    except ValueError as e:
        return f"I couldn't run that simulation: {e}", None
    except Exception as e:
        return f"Something went wrong while running the simulation: {e}", None


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
