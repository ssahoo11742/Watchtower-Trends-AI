import yfinance as yf
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

# ========== SETTINGS ==========
TICKER_FILE = "tickers.txt"  # One ticker per line
OUTPUT_FILE = "companies.csv"
MAX_KEYWORDS = 20

# ========== STEP 1: READ TICKERS ==========
with open(TICKER_FILE, "r") as f:
    tickers = [line.strip().upper() for line in f if line.strip()]

print(f"Loaded {len(tickers)} tickers...")

# ========== STEP 2: FETCH COMPANY INFO ==========
data = []

for i, ticker in enumerate(tickers, 1):
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ""
        desc = info.get("longBusinessSummary") or ""
        if not desc:
            print(f"[{ticker}] No description found.")
            continue
        data.append({"Ticker": ticker, "Name": name, "Description": desc})
        print(f"[{i}/{len(tickers)}] Collected {ticker} - {name}")
        time.sleep(1)  # avoid rate limit
    except Exception as e:
        print(f"[{ticker}] Error: {e}")
        time.sleep(1)

df = pd.DataFrame(data)
if df.empty:
    print("No data collected! Check ticker list or internet connection.")
    exit()

# ========== STEP 3: CLEAN TEXT ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\b\d+\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_desc"] = df["Description"].apply(clean_text)

# ========== STEP 4: BUILD TF-IDF MODEL ==========
print("Building TF-IDF model...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df["clean_desc"])
feature_names = np.array(vectorizer.get_feature_names_out())

# ========== STEP 5: EXTRACT TOP KEYWORDS ==========
def extract_keywords(row_idx, n=MAX_KEYWORDS):
    row = X[row_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:n]
    keywords = [feature_names[i] for i in top_indices if len(feature_names[i]) > 2]
    return ", ".join(keywords)

df["Keywords"] = [extract_keywords(i) for i in range(len(df))]

# ========== STEP 6: SAVE ==========
df[["Ticker", "Name", "Description", "Keywords"]].to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} companies to {OUTPUT_FILE}")
