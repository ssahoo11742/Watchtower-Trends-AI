from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from functools import lru_cache
import yfinance as yf
# Initialize Alpaca API
api = REST(
    key_id='PKA3IISWCD7DRNCGCTLXDWNQ3N',
    secret_key='9Z33gJtEoGG8CUfpLcuH5CjK2QwSLPBCsjxqefL2WjpG'
)

# Financial Modeling Prep API key (get free key at https://financialmodelingprep.com/developer/docs/)
FMP_API_KEY = "yBi4eA9XJO74nk4T1MVi2hlnmJI5DOo0"  # Replace with your key
USE_FMP = True  # Set to True once you have an API key

import time
from datetime import datetime, timedelta
from functools import lru_cache
import yfinance as yf

# Rate limiting for yfinance
_last_yf_call = {}
_yf_min_interval = 0.5  # Minimum seconds between calls

def rate_limited_yf_call(ticker):
    """Rate-limited wrapper for yfinance calls"""
    global _last_yf_call
    
    current_time = time.time()
    if ticker in _last_yf_call:
        time_since_last = current_time - _last_yf_call[ticker]
        if time_since_last < _yf_min_interval:
            time.sleep(_yf_min_interval - time_since_last)
    
    _last_yf_call[ticker] = time.time()
    return yf.Ticker(ticker)

# Cache with longer TTL (1 hour = 3600 seconds)
_fundamentals_cache = {}
_cache_ttl = 3600  # 1 hour

def fetch_fundamentals(ticker):
    """Fetch fundamental data from Yahoo Finance with caching and rate limiting"""
    global _fundamentals_cache
    
    # Check cache first
    cache_key = ticker.upper()
    if cache_key in _fundamentals_cache:
        cached_data, timestamp = _fundamentals_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            print(f"Using cached fundamentals for {ticker} (age: {int(time.time() - timestamp)}s)")
            return cached_data
    
    try:
        print(f"Fetching fundamentals for {ticker} from Yahoo Finance...")
        
        # Rate-limited call to yfinance
        stock = rate_limited_yf_call(ticker)
        info = stock.info
        
        # Check if we got valid data
        if not info or len(info) == 0:
            print(f"  No data returned from Yahoo Finance for {ticker}")
            return {}
        
        fundamentals = {}
        
        # Helper function to safely get values
        def safe_get(key, default=None, multiplier=1):
            try:
                value = info.get(key)
                if value is None:
                    return default
                # Handle string 'N/A' or empty values
                if isinstance(value, str) and value.upper() in ['N/A', '', 'NONE']:
                    return default
                return float(value) * multiplier if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Valuation metrics
        fundamentals['pe_ratio'] = safe_get('trailingPE') or safe_get('forwardPE')
        fundamentals['peg_ratio'] = safe_get('pegRatio')
        fundamentals['price_to_book'] = safe_get('priceToBook')
        fundamentals['market_cap'] = safe_get('marketCap', 0)
        
        # Profitability metrics
        fundamentals['profit_margin'] = safe_get('profitMargins')
        fundamentals['roe'] = safe_get('returnOnEquity')
        
        # Financial health
        fundamentals['debt_to_equity'] = safe_get('debtToEquity')
        
        # Growth metrics
        fundamentals['revenue_growth'] = safe_get('revenueGrowth')
        fundamentals['earnings_growth'] = safe_get('earningsGrowth') or safe_get('earningsQuarterlyGrowth')
        
        # Dividend and risk
        div_yield = safe_get('dividendYield')
        if div_yield and div_yield > 0:
            fundamentals['dividend_yield'] = div_yield * 100  # Convert to percentage
        else:
            fundamentals['dividend_yield'] = 0
            
        fundamentals['beta'] = safe_get('beta')
        
        # Cache the result
        _fundamentals_cache[cache_key] = (fundamentals, time.time())
        
        # Log what we found
        print(f"  Yahoo Finance data extracted:")
        print(f"    PE Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
        print(f"    PEG Ratio: {fundamentals.get('peg_ratio', 'N/A')}")
        print(f"    Price/Book: {fundamentals.get('price_to_book', 'N/A')}")
        print(f"    Profit Margin: {fundamentals.get('profit_margin', 'N/A')}")
        print(f"    ROE: {fundamentals.get('roe', 'N/A')}")
        print(f"    Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}")
        print(f"    Revenue Growth: {fundamentals.get('revenue_growth', 'N/A')}")
        print(f"    Earnings Growth: {fundamentals.get('earnings_growth', 'N/A')}")
        print(f"    Dividend Yield: {fundamentals.get('dividend_yield', 0):.2f}%")
        print(f"    Beta: {fundamentals.get('beta', 'N/A')}")
        
        return fundamentals
        
    except Exception as e:
        print(f"Yahoo Finance fundamentals fetch error for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty dict with all keys set to None/0
        return {
            'pe_ratio': None,
            'peg_ratio': None,
            'price_to_book': None,
            'market_cap': 0,
            'profit_margin': None,
            'roe': None,
            'debt_to_equity': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'dividend_yield': 0,
            'beta': None
        }
# =====================================
# COMPREHENSIVE STOCK DATA FETCHING
# ============================================================================

def fetch_comprehensive_stock_data(ticker):
    """Fetch all data needed for multi-timeframe analysis"""
    try:
        # Calculate date ranges
        end_date = datetime.now()
        start_1mo = end_date - timedelta(days=30)
        start_3mo = end_date - timedelta(days=90)
        start_1y = end_date - timedelta(days=365)
        start_5y = end_date - timedelta(days=1825)
        
        # Get different timeframe histories (using IEX feed for free tier)
        bars_1mo = api.get_bars(ticker, "1Day", start=start_1mo.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), feed='iex').df
        bars_3mo = api.get_bars(ticker, "1Day", start=start_3mo.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), feed='iex').df
        bars_1y = api.get_bars(ticker, "1Day", start=start_1y.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), feed='iex').df
        bars_5y = api.get_bars(ticker, "1Day", start=start_5y.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), feed='iex').df
        
        if bars_1mo.empty:
            return None
        
        # Rename columns to match yfinance format
        hist_1mo = bars_1mo.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        hist_3mo = bars_3mo.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        hist_1y = bars_1y.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        hist_5y = bars_5y.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        
        # Get asset info (limited data available from Alpaca)
        try:
            asset = api.get_asset(ticker)
        except:
            asset = None
        
        # Current metrics
        current_price = hist_1mo['Close'].iloc[-1]
        
        # Short-term metrics (1 day to 1 month)
        price_1d_ago = hist_1mo['Close'].iloc[-2] if len(hist_1mo) >= 2 else current_price
        price_1w_ago = hist_1mo['Close'].iloc[-5] if len(hist_1mo) >= 5 else current_price
        price_1m_ago = hist_1mo['Close'].iloc[0]
        
        change_1d = ((current_price - price_1d_ago) / price_1d_ago) * 100
        change_1w = ((current_price - price_1w_ago) / price_1w_ago) * 100
        change_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
        
        # Medium-term metrics (3 months)
        price_3m_ago = hist_3mo['Close'].iloc[0] if not hist_3mo.empty else current_price
        change_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
        
        # Long-term metrics (1 year, 5 years)
        price_1y_ago = hist_1y['Close'].iloc[0] if not hist_1y.empty else current_price
        change_1y = ((current_price - price_1y_ago) / price_1y_ago) * 100
        
        price_5y_ago = hist_5y['Close'].iloc[0] if len(hist_5y) > 0 else current_price
        change_5y = ((current_price - price_5y_ago) / price_5y_ago) * 100 if len(hist_5y) > 0 else 0
        
        # Volume analysis
        avg_volume_5d = hist_1mo['Volume'].tail(5).mean()
        avg_volume_30d = hist_1mo['Volume'].mean()
        current_volume = hist_1mo['Volume'].iloc[-1]
        volume_spike_ratio = current_volume / avg_volume_5d if avg_volume_5d > 0 else 1.0
        
        # Volatility (different timeframes)
        volatility_1w = hist_1mo['Close'].tail(5).pct_change().std() * 100 if len(hist_1mo) >= 5 else 0
        volatility_1m = hist_1mo['Close'].pct_change().std() * 100
        volatility_3m = hist_3mo['Close'].pct_change().std() * 100 if not hist_3mo.empty else volatility_1m
        volatility_1y = hist_1y['Close'].pct_change().std() * 100 if not hist_1y.empty else volatility_3m
        
        # Calculate Beta (vs SPY market)
        beta = calculate_beta(ticker, hist_1y)
        
        # Moving averages
        ma_10 = hist_1mo['Close'].tail(10).mean() if len(hist_1mo) >= 10 else current_price
        ma_20 = hist_1mo['Close'].tail(20).mean() if len(hist_1mo) >= 20 else current_price
        ma_50 = hist_3mo['Close'].tail(50).mean() if len(hist_3mo) >= 50 else current_price
        ma_200 = hist_1y['Close'].tail(200).mean() if len(hist_1y) >= 200 else current_price
        
        # RSI (different periods)
        rsi_14 = calculate_rsi(hist_1mo['Close'], 14)
        rsi_7 = calculate_rsi(hist_1mo['Close'], 7) if len(hist_1mo) >= 7 else rsi_14
        
        # 52-week range
        high_52w = hist_1y['High'].max() if not hist_1y.empty else current_price
        low_52w = hist_1y['Low'].min() if not hist_1y.empty else current_price
        price_position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50
        
        # Fundamental metrics (Alpaca doesn't provide these, set to None)
        # Fetch fundamental metrics from FMP
        fundamentals = fetch_fundamentals(ticker)
        print(fundamentals)
        pe_ratio = fundamentals.get('pe_ratio')
        forward_pe = None  # Not provided by FMP in current implementation
        peg_ratio = fundamentals.get('peg_ratio')
        price_to_book = fundamentals.get('price_to_book')
        profit_margin = fundamentals.get('profit_margin')
        roe = fundamentals.get('roe')
        debt_to_equity = fundamentals.get('debt_to_equity')
        revenue_growth = fundamentals.get('revenue_growth')
        earnings_growth = fundamentals.get('earnings_growth')
        market_cap = fundamentals.get('market_cap', 0)
        dividend_yield = fundamentals.get('dividend_yield', 0)
        # Beta already calculated above from price data
        # Beta calculated above
        
        return {
            # Price data
            'ticker': ticker,
            'current_price': current_price,
            'change_1d': change_1d,
            'change_1w': change_1w,
            'change_1m': change_1m,
            'change_3m': change_3m,
            'change_1y': change_1y,
            'change_5y': change_5y,
            
            # Volume
            'current_volume': current_volume,
            'avg_volume_5d': avg_volume_5d,
            'avg_volume_30d': avg_volume_30d,
            'volume_spike_ratio': volume_spike_ratio,
            
            # Volatility
            'volatility_1w': volatility_1w,
            'volatility_1m': volatility_1m,
            'volatility_3m': volatility_3m,
            'volatility_1y': volatility_1y,
            'beta': beta,
            
            # Moving averages
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'price_vs_ma10': ((current_price - ma_10) / ma_10) * 100,
            'price_vs_ma20': ((current_price - ma_20) / ma_20) * 100,
            'price_vs_ma50': ((current_price - ma_50) / ma_50) * 100,
            'price_vs_ma200': ((current_price - ma_200) / ma_200) * 100,
            
            # Technical indicators
            'rsi_7': rsi_7,
            'rsi_14': rsi_14,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'price_position_52w': price_position_52w,
            
            # Fundamentals
            'pe_ratio': pe_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'profit_margin': profit_margin,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'market_cap': market_cap,
            'dividend_yield': dividend_yield
        }
        
    except Exception as e:
        print(e)
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0

def calculate_beta(ticker, hist_stock):
    """Calculate Beta vs SPY (market)"""
    try:
        if len(hist_stock) < 30:
            return None
        
        # Get SPY data for same period
        start_date = hist_stock.index[0].strftime('%Y-%m-%d')
        end_date = hist_stock.index[-1].strftime('%Y-%m-%d')
        spy_bars = api.get_bars('SPY', '1Day', start=start_date, end=end_date, feed='iex').df
        
        if spy_bars.empty:
            return None
        
        # Align dates
        spy_bars = spy_bars.rename(columns={'close': 'Close'})
        stock_returns = hist_stock['Close'].pct_change().dropna()
        spy_returns = spy_bars['Close'].pct_change().dropna()
        
        # Find common dates
        common_dates = stock_returns.index.intersection(spy_returns.index)
        if len(common_dates) < 30:
            return None
        
        stock_returns = stock_returns.loc[common_dates]
        spy_returns = spy_returns.loc[common_dates]
        
        # Calculate beta: Covariance(stock, market) / Variance(market)
        covariance = np.cov(stock_returns, spy_returns)[0][1]
        market_variance = np.var(spy_returns)
        
        if market_variance == 0:
            return None
        
        beta = covariance / market_variance
        return float(beta)
        
    except Exception as e:
        print(f"Beta calculation error: {e}")
        return None

def normalize_score(value, min_val, max_val, invert=False):
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
    return 1 - normalized if invert else normalized

# ============================================================================
# MULTI-TIMEFRAME SCORING FUNCTIONS
# ============================================================================

def calculate_day_trader_score(data):
    """Score for day trading (1d-1w)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. IMMEDIATE MOMENTUM (40%)
    momentum_1d = normalize_score(data['change_1d'], -3, 3)
    momentum_1w = normalize_score(data['change_1w'], -8, 8)
    scores['momentum'] = (momentum_1d * 0.6) + (momentum_1w * 0.4)
    
    # 2. VOLUME SPIKE (35%)
    volume_score = normalize_score(data['volume_spike_ratio'], 0.8, 3.0)
    scores['volume'] = volume_score
    
    # 3. INTRADAY VOLATILITY (25%)
    vol_score = normalize_score(data['volatility_1w'], 0, 8)
    if data['volatility_1w'] > 10:
        vol_score *= 0.5
    scores['volatility'] = vol_score
    
    # Bonus for price above 10-day MA
    if data['price_vs_ma10'] > 0:
        scores['momentum'] = min(1.0, scores['momentum'] * 1.15)
    
    weights = {'momentum': 0.40, 'volume': 0.35, 'volatility': 0.25}
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    # Categorize
    if composite > 0.75 and data['volume_spike_ratio'] > 2.0:
        category = "🚀 HOT MOMENTUM"
    elif composite > 0.65:
        category = "📈 TRENDING"
    elif composite > 0.45:
        category = "⚡ VOLATILE"
    elif composite > 0.30:
        category = "📊 CHOPPY"
    else:
        category = "🛑 AVOID"
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': category
    }

def calculate_swing_trader_score(data):
    """Score for swing trading (1w-3m)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. TREND MOMENTUM (35%)
    momentum_raw = (
        data['change_1w'] * 0.3 +
        data['change_1m'] * 0.4 +
        data['change_3m'] * 0.3
    )
    scores['momentum'] = normalize_score(momentum_raw, -15, 15)
    
    # 2. MOVING AVERAGE ALIGNMENT (30%)
    ma_score = 0
    if data['price_vs_ma20'] > 0:
        ma_score += 0.35
    if data['price_vs_ma50'] > 0:
        ma_score += 0.35
    if data['ma_20'] > data['ma_50']:
        ma_score += 0.30
    scores['trend_alignment'] = ma_score
    
    # 3. VOLUME CONFIRMATION (20%)
    volume_score = normalize_score(data['volume_spike_ratio'], 0.7, 2.0)
    scores['volume'] = volume_score
    
    # 4. RSI & POSITION (15%)
    rsi = data['rsi_14']
    if 40 <= rsi <= 60:
        rsi_score = 1.0
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        rsi_score = 0.7
    else:
        rsi_score = 0.3
    
    position = data['price_position_52w']
    if 30 <= position <= 70:
        position_score = 1.0
    elif position < 30:
        position_score = 0.6
    else:
        position_score = 0.4
    
    scores['rsi_position'] = (rsi_score + position_score) / 2
    
    weights = {
        'momentum': 0.35,
        'trend_alignment': 0.30,
        'volume': 0.20,
        'rsi_position': 0.15
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    # Categorize
    if composite > 0.75 and data['price_vs_ma50'] > 0:
        category = "🎯 STRONG UPTREND"
    elif composite > 0.65:
        category = "📈 BULLISH SETUP"
    elif composite > 0.50:
        category = "👀 CONSOLIDATING"
    elif composite > 0.35:
        category = "⚠️ MIXED SIGNALS"
    else:
        category = "🔻 DOWNTREND"
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': category
    }

def calculate_position_trader_score(data):
    """Score for position trading (3m-1y)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. MEDIUM-TERM MOMENTUM (30%)
    momentum_raw = (
        data['change_3m'] * 0.5 +
        data['change_1y'] * 0.5
    )
    scores['momentum'] = normalize_score(momentum_raw, -20, 30)
    
    # 2. LONG-TERM TREND (25%)
    if data['price_vs_ma200'] > 5:
        trend_score = 1.0
    elif data['price_vs_ma200'] > 0:
        trend_score = 0.8
    elif data['price_vs_ma200'] > -5:
        trend_score = 0.5
    else:
        trend_score = 0.2
    
    if data['ma_50'] > data['ma_200']:
        trend_score = min(1.0, trend_score * 1.2)
    
    scores['long_term_trend'] = trend_score
    
    # 3. VALUATION BASICS (25%)
    valuation_score = 0.5
    
    if data['pe_ratio'] and data['pe_ratio'] > 0:
        if data['pe_ratio'] < 20:
            valuation_score += 0.3
        elif data['pe_ratio'] > 40:
            valuation_score -= 0.2
    
    if data['peg_ratio'] and data['peg_ratio'] > 0:
        if data['peg_ratio'] < 1.5:
            valuation_score += 0.2
        elif data['peg_ratio'] > 2.5:
            valuation_score -= 0.2
    
    scores['valuation'] = max(0, min(1, valuation_score))
    
    # 4. STABILITY (20%)
    vol_score = normalize_score(data['volatility_3m'], 0, 8, invert=True)
    scores['stability'] = vol_score
    
    weights = {
        'momentum': 0.30,
        'long_term_trend': 0.25,
        'valuation': 0.25,
        'stability': 0.20
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    # Categorize
    if composite > 0.75:
        category = "💎 STRONG BUY"
    elif composite > 0.65:
        category = "✅ BUY"
    elif composite > 0.50:
        category = "👍 HOLD"
    elif composite > 0.35:
        category = "⚠️ CAUTION"
    else:
        category = "❌ SELL/AVOID"
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': category
    }

def calculate_longterm_investor_score(data):
    """Score for long-term investing (1y+)"""
    if not data:
        return None
    
    scores = {}
    
    # Check if we have fundamental data
    has_fundamentals = any([
        data['pe_ratio'],
        data['peg_ratio'],
        data['profit_margin'],
        data['roe'],
        data['revenue_growth'],
        data['earnings_growth']
    ])
    
    if has_fundamentals:
        # Original scoring with fundamentals
        # 1. VALUATION (30%)
        valuation_score = 0.5
        
        if data['pe_ratio'] and data['pe_ratio'] > 0:
            if data['pe_ratio'] < 15:
                valuation_score += 0.25
            elif data['pe_ratio'] < 25:
                valuation_score += 0.10
            elif data['pe_ratio'] > 50:
                valuation_score -= 0.25
        
        if data['peg_ratio'] and data['peg_ratio'] > 0:
            if data['peg_ratio'] < 1.0:
                valuation_score += 0.25
            elif data['peg_ratio'] < 2.0:
                valuation_score += 0.10
            elif data['peg_ratio'] > 3.0:
                valuation_score -= 0.20
        
        if data['price_to_book'] and data['price_to_book'] > 0:
            if data['price_to_book'] < 3:
                valuation_score += 0.10
            elif data['price_to_book'] > 10:
                valuation_score -= 0.10
        
        scores['valuation'] = max(0, min(1, valuation_score))
        
        # 2. QUALITY & PROFITABILITY (25%)
        quality_score = 0.5
        
        if data['profit_margin'] and data['profit_margin'] > 0:
            if data['profit_margin'] > 0.20:
                quality_score += 0.25
            elif data['profit_margin'] > 0.10:
                quality_score += 0.10
        
        if data['roe'] and data['roe'] > 0:
            if data['roe'] > 0.15:
                quality_score += 0.25
            elif data['roe'] > 0.10:
                quality_score += 0.10
        
        scores['quality'] = max(0, min(1, quality_score))
        
        # 3. GROWTH (25%)
        growth_score = 0.5
        
        if data['revenue_growth'] and data['revenue_growth'] > 0:
            if data['revenue_growth'] > 0.15:
                growth_score += 0.25
            elif data['revenue_growth'] > 0.05:
                growth_score += 0.15
            elif data['revenue_growth'] < -0.05:
                growth_score -= 0.20
        
        if data['earnings_growth'] and data['earnings_growth'] > 0:
            if data['earnings_growth'] > 0.15:
                growth_score += 0.25
            elif data['earnings_growth'] > 0.05:
                growth_score += 0.15
        
        scores['growth'] = max(0, min(1, growth_score))
        
        # 4. FINANCIAL HEALTH & INCOME (20%)
        health_score = 0.5
        
        if data['debt_to_equity'] is not None:
            if data['debt_to_equity'] < 50:
                health_score += 0.25
            elif data['debt_to_equity'] > 150:
                health_score -= 0.25
        
        if data['dividend_yield'] and data['dividend_yield'] > 0:
            if data['dividend_yield'] > 3:
                health_score += 0.25
            elif data['dividend_yield'] > 1.5:
                health_score += 0.15
        
        scores['health_income'] = max(0, min(1, health_score))
        
        weights = {
            'valuation': 0.30,
            'quality': 0.25,
            'growth': 0.25,
            'health_income': 0.20
        }
    else:
        # Alternative scoring without fundamentals (technical-based)
        # 1. LONG-TERM PERFORMANCE (40%)
        perf_1y = normalize_score(data['change_1y'], -20, 50)
        perf_5y = normalize_score(data['change_5y'], -50, 200)
        scores['long_term_performance'] = (perf_1y * 0.6) + (perf_5y * 0.4)
        
        # 2. TREND STRENGTH (30%)
        if data['price_vs_ma200'] > 10:
            trend_score = 1.0
        elif data['price_vs_ma200'] > 0:
            trend_score = 0.7
        elif data['price_vs_ma200'] > -10:
            trend_score = 0.4
        else:
            trend_score = 0.2
        
        if data['ma_50'] > data['ma_200']:
            trend_score = min(1.0, trend_score * 1.2)
        
        scores['trend_strength'] = trend_score
        
        # 3. STABILITY (20%)
        vol_score = normalize_score(data['volatility_1y'], 0, 10, invert=True)
        scores['stability'] = vol_score
        
        # 4. RELATIVE POSITION (10%)
        position_score = normalize_score(data['price_position_52w'], 20, 80)
        scores['relative_position'] = position_score
        
        weights = {
            'long_term_performance': 0.40,
            'trend_strength': 0.30,
            'stability': 0.20,
            'relative_position': 0.10
        }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    # Categorize
    if composite > 0.75:
        category = "⭐ EXCELLENT"
    elif composite > 0.65:
        category = "👍 GOOD"
    elif composite > 0.50:
        category = "✋ FAIR"
    elif composite > 0.35:
        category = "⚠️ BELOW AVERAGE"
    else:
        category = "🚫 POOR"
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': category
    }