# whatif.py - All what-if simulation functions for the Optima DSS

import pandas as pd
from config import FEATURE_COLS, MONTH_NAMES, VALID_DISCOUNTS, FIXED_STORE_ID
from data_loader import load_all

# Load shared resources 
train_data, model, PRODUCT_AVG_PRICE, PR_MIN, PR_MAX = load_all()


# Core Prediction
def predict_from_row(row):
    """Takes a modified row dict and returns predicted Weekly_Sales in SAR."""
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    input_df = pd.DataFrame([row])[FEATURE_COLS]
    return float(model.predict(input_df)[0])


# Baseline Builders
def get_baseline(store_id, product_id, month, year=None):
    """
    Finds the latest real row for this store/product/month.
    If year is not specified, uses the most recent year available.
    Returns (row_dict, predicted_sales).
    """
    match = train_data[
        (train_data['Store ID']   == store_id) &
        (train_data['Product ID'] == product_id) &
        (train_data['Month']      == month)
    ]

    if len(match) == 0:
        raise ValueError(
            f"No data found for Product {product_id} "
            f"in Store {store_id}, Month {month}."
        )

    if year is None:
        year = int(match['Year'].max())

    match = match[match['Year'] == year]

    if len(match) == 0:
        raise ValueError(
            f"No data found for Product {product_id} "
            f"in Store {store_id}, Month {month}, Year {year}."
        )

    row = match.sort_values('Week_Start', ascending=False).iloc[0].to_dict()

    row['Month']      = month
    row['Year']       = year
    row['WeekOfYear'] = pd.Timestamp(
        year=year, month=month, day=15
    ).isocalendar()[1]

    avg_price = PRODUCT_AVG_PRICE.get(product_id, train_data['Avg_Price'].mean())
    row['Price_Relative'] = row['Avg_Price'] / avg_price if avg_price > 0 else 1.0

    pred = float(model.predict(pd.DataFrame([row])[FEATURE_COLS])[0])
    return row, pred


def get_latest_baseline(store_id, product_id):
    """
    Returns the absolute latest row for a product in a store.
    No month filtering — uses the most recent week available.
    Used by chatbot when user does not specify a month.
    """
    match = train_data[
        (train_data['Store ID']   == store_id) &
        (train_data['Product ID'] == product_id)
    ]

    if len(match) == 0:
        raise ValueError(
            f"No data found for Product {product_id} in Store {store_id}."
        )

    row = match.sort_values('Week_Start', ascending=False).iloc[0].to_dict()

    avg_price = PRODUCT_AVG_PRICE.get(product_id, train_data['Avg_Price'].mean())
    row['Price_Relative'] = row['Avg_Price'] / avg_price if avg_price > 0 else 1.0

    pred = float(model.predict(pd.DataFrame([row])[FEATURE_COLS])[0])
    return row, pred


# Validation
def validate_scenario(row):
    """
    Validates scenario inputs before prediction.
    Raises ValueError for hard errors.
    Prints warning for soft issues.
    """
    if row['Avg_Price'] <= 0:
        raise ValueError("Avg_Price must be greater than 0.")

    if not (0 <= row['Avg_Discount'] <= 0.45):
        raise ValueError("Avg_Discount must be between 0.00 and 0.45.")

    if not (0 <= row['Campaign_Discount'] <= 0.45):
        raise ValueError("Campaign_Discount must be between 0.00 and 0.45.")

    if row['Avg_Price'] <= row['Production Cost']:
        print(
            f"Warning: Avg_Price (SAR {row['Avg_Price']:.2f}) "
            f"is below Production Cost (SAR {row['Production Cost']:.2f})."
        )

    if not (PR_MIN <= row['Price_Relative'] <= PR_MAX):
        print(
            f"Warning: Price_Relative ({row['Price_Relative']:.3f}) "
            f"is outside training range [{PR_MIN:.3f}, {PR_MAX:.3f}]."
        )

    return True


# What-If Scenarios

# 1: price change
def whatif_price_change(baseline, baseline_pred, new_price_increase):
    """
    Simulates increasing the selling price by a fixed SAR amount.
    Automatically recomputes Price_Relative.
    Returns structured result dict.
    """
    from config import PRODUCT_NAMES
    row            = baseline.copy()
    original_price = row['Avg_Price']
    final_price    = original_price + new_price_increase

    row['Avg_Price']      = final_price
    avg_price             = PRODUCT_AVG_PRICE.get(
        row['Product ID'], train_data['Avg_Price'].mean()
    )
    row['Price_Relative'] = final_price / avg_price if avg_price > 0 else 1.0

    validate_scenario(row)

    new_pred  = predict_from_row(row)
    delta     = new_pred - baseline_pred
    delta_pct = (delta / baseline_pred) * 100 if baseline_pred > 0 else 0

    return {
        'scenario':       'price_change',
        'product_id':     int(row['Product ID']),
        'product_name':   PRODUCT_NAMES.get(int(row['Product ID']), f"Product {int(row['Product ID'])}"),
        'store_id':       int(row['Store ID']),
        'old_price':      round(original_price, 2),
        'new_price':      round(final_price, 2),
        'price_increase': round(new_price_increase, 2),
        'baseline_sales': round(baseline_pred, 2),
        'new_sales':      round(new_pred, 2),
        'difference':     round(delta, 2),
        'difference_pct': round(delta_pct, 1),
    }


# 2: discount change
def whatif_avg_discount(baseline, baseline_pred, new_discount):
    """
    Simulates changing the discount rate.
    Updates Avg_Price and Price_Relative automatically.
    Returns structured result dict.
    """
    from config import PRODUCT_NAMES
    if new_discount not in VALID_DISCOUNTS:
        raise ValueError(f"Discount must be one of: {VALID_DISCOUNTS}")

    row               = baseline.copy()
    original_discount = row['Avg_Discount']
    original_price    = row['Avg_Price']

    row['Avg_Discount']   = new_discount
    row['Avg_Price']      = original_price * (1 - new_discount)
    avg_price             = PRODUCT_AVG_PRICE.get(
        row['Product ID'], train_data['Avg_Price'].mean()
    )
    row['Price_Relative'] = row['Avg_Price'] / avg_price if avg_price > 0 else 1.0

    validate_scenario(row)

    new_pred  = predict_from_row(row)
    delta     = new_pred - baseline_pred
    delta_pct = (delta / baseline_pred) * 100 if baseline_pred > 0 else 0

    return {
        'scenario':         'discount_change',
        'product_id':       int(row['Product ID']),
        'product_name':     PRODUCT_NAMES.get(int(row['Product ID']), f"Product {int(row['Product ID'])}"),
        'store_id':         int(row['Store ID']),
        'old_discount_pct': round(original_discount * 100, 1),
        'new_discount_pct': round(new_discount * 100, 1),
        'old_price':        round(original_price, 2),
        'new_price':        round(row['Avg_Price'], 2),
        'baseline_sales':   round(baseline_pred, 2),
        'new_sales':        round(new_pred, 2),
        'difference':       round(delta, 2),
        'difference_pct':   round(delta_pct, 1),
    }


# 2: discount extension
def whatif_extended_discount(store_id, product_id, start_month, end_month, new_discount):
    """
    Simulates running a discount across multiple consecutive months.
    Returns structured result dict with status, message, and monthly breakdown.
    """
    from config import PRODUCT_NAMES
    if new_discount not in VALID_DISCOUNTS:
        raise ValueError(f"Discount must be one of: {VALID_DISCOUNTS}")

    rows           = []
    missing_months = []

    for month in range(start_month, end_month + 1):
        try:
            b, b_pred      = get_baseline(store_id, product_id, month)
            row            = b.copy()
            original_price = row['Avg_Price']

            row['Avg_Discount']   = new_discount
            row['Avg_Price']      = original_price * (1 - new_discount)
            avg_price             = PRODUCT_AVG_PRICE.get(
                product_id, train_data['Avg_Price'].mean()
            )
            row['Price_Relative'] = row['Avg_Price'] / avg_price if avg_price > 0 else 1.0

            validate_scenario(row)

            new_pred  = predict_from_row(row)
            delta     = new_pred - b_pred
            delta_pct = (delta / b_pred) * 100 if b_pred > 0 else 0

            rows.append({
                'month':          MONTH_NAMES[month],
                'baseline_sales': round(b_pred, 2),
                'new_sales':      round(new_pred, 2),
                'delta_sar':      round(delta, 2),
                'delta_pct':      round(delta_pct, 1),
            })

        except ValueError as e:
            if 'No data found' in str(e):
                missing_months.append(MONTH_NAMES[month])
            else:
                raise

    if len(rows) == 0:
        return {
            'status':         'no_data',
            'message':        'No historical data found for the selected period.',
            'missing_months': missing_months,
            'monthly_detail': [],
            'total':          None,
        }

    total_baseline  = sum(r['baseline_sales'] for r in rows)
    total_new       = sum(r['new_sales']      for r in rows)
    total_delta     = total_new - total_baseline
    total_delta_pct = (total_delta / total_baseline) * 100 if total_baseline > 0 else 0

    status  = 'success' if not missing_months else 'partial_data'
    message = (
        'Simulation completed successfully for all selected months.'
        if not missing_months else
        f"Simulation completed. Missing months excluded: {', '.join(missing_months)}."
    )

    return {
        'scenario':       'extended_discount',
        'product_id':     product_id,
        'product_name':   PRODUCT_NAMES.get(product_id, f"Product {product_id}"),
        'store_id':       store_id,
        'discount_pct':   round(new_discount * 100, 1),
        'start_month':    MONTH_NAMES[start_month],
        'end_month':      MONTH_NAMES[end_month],
        'status':         status,
        'message':        message,
        'missing_months': missing_months,
        'monthly_detail': rows,
        'total': {
            'baseline_sales': round(total_baseline, 2),
            'new_sales':      round(total_new, 2),
            'delta_sar':      round(total_delta, 2),
            'delta_pct':      round(total_delta_pct, 1),
        }
    }