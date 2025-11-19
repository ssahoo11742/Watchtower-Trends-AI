import pandas as pd
import yfinance as yf
import time

INPUT_FILE = "./data/companies.csv"
OUTPUT_FILE = "companies_extended.csv"

def save_progress(df, sectors, industries, market_caps, output_file, start_index):
    """Save current progress to file"""
    df_temp = df.copy()
    total_processed = len(sectors)
    df_temp = df_temp.iloc[:total_processed].copy()
    df_temp["Sector"] = sectors
    df_temp["Industry"] = industries
    df_temp["MarketCap"] = market_caps
    df_temp.to_csv(output_file, index=False)
    print(f"\n✓ Progress saved to {output_file}")
    print(f"  Records processed: {total_processed}")
    print(f"  Records with sector: {df_temp['Sector'].notna().sum()}")
    print(f"  Records with industry: {df_temp['Industry'].notna().sum()}")
    print(f"  Records with market cap: {df_temp['MarketCap'].notna().sum()}")

def extend_company_csv(input_file, output_file, start_index=None):
    # Read CSV
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} companies from {input_file}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        return
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return
    
    # Check if Ticker column exists
    if "Ticker" not in df.columns:
        print(f"ERROR: 'Ticker' column not found. Available columns: {df.columns.tolist()}")
        return
    
    # Auto-detect where to resume from if not specified
    if start_index is None:
        try:
            df_existing = pd.read_csv(output_file)
            if "Sector" in df_existing.columns:
                start_index = len(df_existing)
                print(f"📂 Found existing file with {start_index} rows. Resuming from index {start_index}")
                sectors = df_existing["Sector"].tolist()
                industries = df_existing["Industry"].tolist()
                market_caps = df_existing["MarketCap"].tolist()
            else:
                start_index = 0
                sectors = []
                industries = []
                market_caps = []
        except FileNotFoundError:
            print("📝 Starting fresh - no existing output file found")
            start_index = 0
            sectors = []
            industries = []
            market_caps = []
    # Manual start_index provided
    elif start_index > 0:
        print(f"Resuming from index {start_index}")
        # Load existing columns if they exist
        try:
            df_existing = pd.read_csv(output_file)
            if "Sector" in df_existing.columns:
                sectors = df_existing["Sector"].tolist()[:start_index]
                industries = df_existing["Industry"].tolist()[:start_index]
                market_caps = df_existing["MarketCap"].tolist()[:start_index]
            else:
                sectors = [None] * start_index
                industries = [None] * start_index
                market_caps = [None] * start_index
        except FileNotFoundError:
            sectors = [None] * start_index
            industries = [None] * start_index
            market_caps = [None] * start_index
    else:
        sectors = []
        industries = []
        market_caps = []
    
    tickers = df["Ticker"].astype(str).tolist()
    
    try:
        for i in range(start_index, len(tickers)):
            t = tickers[i]
            
            # Skip ETFs if Name column exists and contains "ETF"
            if "Name" in df.columns:
                name = str(df.loc[i, "Name"])
                if "ETF" in name.upper():
                    print(f"\n[{i+1}/{len(tickers)}] Skipping {t} (ETF: {name})")
                    sectors.append(None)
                    industries.append(None)
                    market_caps.append(None)
                    continue
            
            print(f"\n[{i+1}/{len(tickers)}] Fetching {t}...")
            try:
                # Create ticker object
                ticker = yf.Ticker(t)
                
                # Get info dictionary
                info = ticker.info
                
                # Check for rate limiting indicators
                if not info or len(info) == 0:
                    raise Exception("Empty response - possible rate limiting")
                
                # Extract values with defaults
                sector = info.get('sector', None)
                industry = info.get('industry', None)
                market_cap = info.get('marketCap', None)
                
                print(f"  ✓ Sector: {sector}")
                print(f"  ✓ Industry: {industry}")
                print(f"  ✓ Market Cap: {market_cap}")
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check for rate limiting errors
                if any(keyword in error_msg for keyword in ['rate limit', '429', 'too many requests', 'empty response']):
                    print(f"  ✗ RATE LIMITED: {e}")
                    print(f"\n⚠ Rate limit detected. Saving progress and exiting...")
                    save_progress(df, sectors, industries, market_caps, output_file, start_index)
                    return
                else:
                    print(f"  ✗ ERROR: {e}")
                    sector = industry = market_cap = None
            
            sectors.append(sector)
            industries.append(industry)
            market_caps.append(market_cap)
            
            # Rate limiting - adjust if needed
            if i < len(tickers) - 1:  # Don't sleep after last ticker
                time.sleep(0.01)  # Increased from 1 to 2 seconds
    
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user. Saving progress...")
        save_progress(df, sectors, industries, market_caps, output_file, start_index)
        return
    
    df_final = df.iloc[:len(sectors)].copy()
    df_final["Sector"] = sectors
    df_final["Industry"] = industries
    df_final["MarketCap"] = market_caps
    
    df_final.to_csv(output_file, index=False)
    print(f"\n✓ Successfully saved to {output_file}")
    print(f"  Records with sector: {df_final['Sector'].notna().sum()}")
    print(f"  Records with industry: {df_final['Industry'].notna().sum()}")
    print(f"  Records with market cap: {df_final['MarketCap'].notna().sum()}")

if __name__ == "__main__":
    extend_company_csv(INPUT_FILE, OUTPUT_FILE)