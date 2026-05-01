# config.py - All constants for the Optima DSS

# Store 
FIXED_STORE_ID = 14

# Products 
PRODUCT_IDS = [8999, 12717, 10013]

PRODUCT_NAMES = {
    8999:  "Wedding Dress",
    12717: "Graduation Dress",
    10013: "Party Dress",
}

# Feature columns
FEATURE_COLS = [
    'Store ID', 'Product ID',
    'Avg_Price', 'Avg_Discount', 'Campaign_Discount',
    'Month', 'WeekOfYear', 'Year',
    'Production Cost', 'Store_Lag1_Total', 'Lag1_Quantity',
    'Price_Relative'
]

# Month names
MONTH_NAMES = {
    1:  'January',   2:  'February', 3:  'March',    4:  'April',
    5:  'May',       6:  'June',     7:  'July',      8:  'August',
    9:  'September', 10: 'October',  11: 'November',  12: 'December'
}

# Valid discount values
VALID_DISCOUNTS = [0.0, 0.25, 0.35, 0.45]

# File paths 
DATA_PATH    = r'G:\My Drive\Optima\Final_clean_data\\'
MODEL_PATH   = r'G:\My Drive\Optima\Optima_model\\'
CHATBOT_PATH = r'G:\My Drive\Optima\Chatbot\\'

TRAIN_DATA_FILE = DATA_PATH  + 'final_train_data.csv'
MODEL_FILE      = MODEL_PATH + 'optima_xgb_model.pkl'
RAG_DOCS_FILE   = CHATBOT_PATH + 'optima_llm_rag_documents.json'

# LLM
OPENAI_MODEL   = 'gpt-4o-mini'
CHROMA_COLLECTION_NAME = 'optima_statistics_rag'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'