import yfinance as yf
import numpy as np
import pandas as pd

def compute_price_corr(t1, t2, period="1y"):
    """Compute price correlation between two tickers."""
    try:
        # Download data
        df1 = yf.download(t1, period=period, progress=False)
        df2 = yf.download(t2, period=period, progress=False)
        
        # Check if dataframes are empty
        if df1.empty or df2.empty:
            print(f"    ⚠️  Empty data for {t1} or {t2}")
            return 0.0
        
        # Extract close price - handle both MultiIndex levels
        def extract_close(df):
            try:
                # Try level 1 first (most common)
                return df.xs('Close', level=1, axis=1).squeeze()
            except (KeyError, IndexError):
                # Fallback to level 0
                try:
                    return df.xs('Close', level=0, axis=1).squeeze()
                except (KeyError, IndexError):
                    # Last resort - direct access
                    return df['Close'].squeeze()
        
        d1 = extract_close(df1)
        d2 = extract_close(df2)
        
        # Align the data by date and drop NaN
        df = pd.DataFrame({'t1': d1, 't2': d2}).dropna()
        
        if len(df) < 20:
            print(f"    ⚠️  Insufficient data: only {len(df)} overlapping days")
            return 0.0
        
        # Calculate returns
        r1 = df['t1'].pct_change().dropna()
        r2 = df['t2'].pct_change().dropna()
        
        if len(r1) < 10 or len(r2) < 10:
            print(f"    ⚠️  Insufficient returns data")
            return 0.0
        
        # Compute correlation
        corr = float(np.corrcoef(r1, r2)[0,1])
        
        if np.isnan(corr):
            print(f"    ⚠️  Correlation is NaN")
            return 0.0
            
        print(f"    ✅ Correlation: {corr:.4f} (from {len(df)} days)")
        return corr
        
    except Exception as e:
        print(f"    ⚠️  Error computing correlation: {e}")
        return 0.0 


def compute_correlation_strength(ticker, edge):
    t1 = ticker
    node_type = edge['type']
    magnitude = edge['magnitude']
    relevance = edge['relevance']

    # 1. Pick comparison ticker depending on node type
    proxy_ticker = None

    if node_type == "sector":
        sector = edge['data'].get('sector_name', None)
        proxy_ticker, score = _sector_to_etf(sector)  # FIX: Unpack tuple
        print(f"  Sector '{sector}' → ETF: {proxy_ticker} (score: {score})")
        if not proxy_ticker:
            print(f"  ⚠️  No ETF match found for sector '{sector}'")

    elif node_type == "industry":
        industry = edge['data'].get('industry_name', None)
        proxy_ticker, score = _industry_to_etf(industry)  # FIX: Unpack tuple
        print(f"  Industry '{industry}' → ETF: {proxy_ticker} (score: {score})")
        if not proxy_ticker:
            print(f"  ⚠️  No ETF match found for industry '{industry}'")

    elif node_type == "commodity":
        commodity = edge['data'].get('commodity_name', None)
        proxy_ticker, score = _commodity_to_etf(commodity)  # FIX: Unpack tuple
        print(f"  Commodity '{commodity}' → ETF: {proxy_ticker} (score: {score})")
        if not proxy_ticker:
            print(f"  ⚠️  No ETF match found for commodity '{commodity}'")

    elif node_type == "country":
        country = edge['data'].get('country_name', None)
        proxy_ticker, score = _country_to_etf(country)
        print(f"  Country '{country}' → ETF: {proxy_ticker} (score: {score})")
        if not proxy_ticker:
            print(f"  ⚠️  No ETF match found for country '{country}'")

    elif node_type == "supplier":
        proxy_ticker = edge['data'].get("supplier_ticker", None)

    # 2. Compute price corr if proxy exists
    price_corr = 0.0
    if proxy_ticker:
        print(f"  Computing correlation between {t1} and {proxy_ticker}...")
        price_corr = compute_price_corr(t1, proxy_ticker)
        print(f"  Price correlation: {price_corr:.4f}")

    # 3. Volatility similarity
    beta1 = yf.Ticker(t1).info.get("beta", 1.0)
    beta2 = yf.Ticker(proxy_ticker).info.get("beta", 1.0) if proxy_ticker else 1.0
    vol_sim = 1 / (1 + abs(beta1 - beta2))
    print(f"  Beta1: {beta1:.2f}, Beta2: {beta2:.2f}, Vol similarity: {vol_sim:.4f}")

    # 4. Structural dependency
    structural = magnitude * relevance
    print(f"  Structural: {structural:.4f}")

    # Final score
    final_score = (
        0.4 * price_corr +
        0.3 * vol_sim +
        0.3 * structural
    )
    print(f"  Final correlation strength: {final_score:.4f}")
    
    return final_score


# -----------------------------
# Load CSV + Extract ETFs
# -----------------------------
df = pd.read_csv("./data/companies.csv")

# ETFs are recognized because their Name contains "ETF"
etfs = df[df['Name'].str.contains("ETF", case=False)].copy()

# Normalize keywords → list of lowercase strings
etfs["KeywordList"] = (
    etfs["Keywords"]
    .fillna("")
    .apply(lambda k: [x.strip().lower() for x in k.split(",")])
)

# -----------------------------
# Mapping helper function
# -----------------------------
def map_to_etf(label, input_name):
    """Returns (ticker, score) tuple"""
    if not input_name:
        return None, 0
    
    target = input_name.lower()
    best_match = None
    best_score = 0

    for _, row in etfs.iterrows():
        score = 0

        # Keyword matching
        for kw in row["KeywordList"]:
            if kw in target:
                score += 2
            if target in kw:
                score += 2
            if len(target) >= 4 and kw.startswith(target[:4]):
                score += 1

        # Name matching
        name = row["Name"].lower()
        if target in name:
            score += 3
        if label == "commodity" and target in name:
            score += 1  # extra boost for commodities

        if score > best_score:
            best_score = score
            best_match = row["Ticker"]

    return best_match, best_score


# -----------------------------
# Convenience wrappers
# -----------------------------
def _sector_to_etf(sector):
    return map_to_etf("sector", sector)

def _industry_to_etf(industry):
    return map_to_etf("industry", industry)

def _commodity_to_etf(commodity):
    return map_to_etf("commodity", commodity)

def _country_to_etf(country):
    return map_to_etf("country", country)


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    result = compute_correlation_strength("QS",     {
      "type": "country",
      "magnitude": 0.3,
      "relevance": 0.8,
      "direction": "company->country",
      "data": {
        "country_name": "United States",
        "relationship": "economic_contributor",
        "impact_type": "employment_tax_innovation",
        "market_cap": 8159779328
      }
    })
    
    print(f"\n✅ Final result: {result:.4f}")