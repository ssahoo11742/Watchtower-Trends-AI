from fastapi import FastAPI
from pipeline.scoring import (
    fetch_comprehensive_stock_data,
    calculate_day_trader_score,
    calculate_swing_trader_score,
    calculate_position_trader_score,
    calculate_longterm_investor_score
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://watchtower-trends.netlify.app/"],  # Your React app
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)



import math

def format_stock_response(data, topic_id=None, topic_keywords=None, company_name=None, 
                         relevance_score=None, mentions=0, mentioned_as=None,
                         day_score=None, swing_score=None, position_score=None, longterm_score=None):
    """
    Transform stock data into API response format
    """
    
    # Helper to safely convert numpy types and handle NaN/Infinity
    def safe_float(value, decimals=2):
        if value is None:
            return None
        try:
            # Handle numpy types
            if hasattr(value, 'item'):
                value = value.item()
            
            # Convert to float
            value = float(value)
            
            # Check for NaN or Infinity
            if math.isnan(value) or math.isinf(value):
                return None
            
            # Round to specified decimals
            return round(value, decimals)
        except (ValueError, TypeError):
            return None
    
    def safe_int(value):
        if value is None:
            return None
        try:
            if hasattr(value, 'item'):
                value = value.item()
            
            value = int(value)
            
            # Check for unreasonable values
            if abs(value) > 1e15:  # Arbitrary large number check
                return None
                
            return value
        except (ValueError, TypeError):
            return None
    
    # Build the response
    response = {
        "Topic_ID": topic_id,
        "Topic_Keywords": topic_keywords,
        "Ticker": data.get('ticker'),
        "Company_Name": company_name,
        "Relevance_Score": safe_float(relevance_score, 3) if relevance_score else None,
        "Mentions": safe_int(mentions),
        "Mentioned_As": mentioned_as,
        
        # Trading scores
        "Day_Trade_Score": safe_float(day_score['score'], 3) if day_score else None,
        "Day_Trade_Rating": day_score['category'] if day_score else None,
        "Swing_Trade_Score": safe_float(swing_score['score'], 3) if swing_score else None,
        "Swing_Trade_Rating": swing_score['category'] if swing_score else None,
        "Position_Trade_Score": safe_float(position_score['score'], 3) if position_score else None,
        "Position_Trade_Rating": position_score['category'] if position_score else None,
        "LongTerm_Score": safe_float(longterm_score['score'], 2) if longterm_score else None,
        "LongTerm_Rating": longterm_score['category'] if longterm_score else None,
        
        # Price data
        "Current_Price": safe_float(data.get('current_price'), 2),
        "Change_1D": safe_float(data.get('change_1d'), 2),
        "Change_1W": safe_float(data.get('change_1w'), 2),
        "Change_1M": safe_float(data.get('change_1m'), 2),
        "Change_3M": safe_float(data.get('change_3m'), 2),
        "Change_1Y": safe_float(data.get('change_1y'), 2),
        
        # Volume & Technical
        "Volume_Spike_Ratio": safe_float(data.get('volume_spike_ratio'), 2),
        "RSI_14": safe_float(data.get('rsi_14'), 1),
        "Price_vs_MA50": safe_float(data.get('price_vs_ma50'), 2),
        "Price_vs_MA200": safe_float(data.get('price_vs_ma200'), 2),
        
        # Fundamentals
        "PE_Ratio": safe_float(data.get('pe_ratio'), 2),
        "PEG_Ratio": safe_float(data.get('peg_ratio'), 2),
        "Dividend_Yield": safe_float(data.get('dividend_yield'), 2),
        "Profit_Margin": safe_float(data.get('profit_margin'), 4),
        "ROE": safe_float(data.get('roe'), 4)
    }
    
    return response


@app.get("/api/ticker/{symbol}")
def get_ticker(symbol: str):
    """Get complete stock analysis"""
    
    # Fetch data
    data = fetch_comprehensive_stock_data(symbol.upper())
    if not data:
        return {"error": f"Could not fetch data for {symbol}"}
    
    # Calculate scores
    day_score = calculate_day_trader_score(data)
    swing_score = calculate_swing_trader_score(data)
    position_score = calculate_position_trader_score(data)
    longterm_score = calculate_longterm_investor_score(data)
    
    # Format and return
    return format_stock_response(
        data=data,
        day_score=day_score,
        swing_score=swing_score,
        position_score=position_score,
        longterm_score=longterm_score
    )