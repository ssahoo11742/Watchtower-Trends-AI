import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
# ============================================================================
# MULTI-TIMEFRAME STOCK SCORING SYSTEM
# ============================================================================
# Separate scores for different trading styles and timeframes
# - Day Trader (1d-1w): Momentum, volatility, volume spikes
# - Swing Trader (1w-3m): Technical patterns, short-term trends
# - Position Trader (3m-1y): Medium-term trends, fundamentals light
# - Long-term Investor (1y+): Fundamentals, growth, value metrics
# ============================================================================

def fetch_comprehensive_stock_data(ticker):
    """Fetch all data needed for multi-timeframe analysis"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get different timeframe histories
        hist_1mo = stock.history(period='1mo')
        hist_3mo = stock.history(period='3mo')
        hist_1y = stock.history(period='1y')
        hist_5y = stock.history(period='5y')
        
        if hist_1mo.empty:
            return None
        
        info = stock.info
        
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
        
        # Fundamental metrics
        pe_ratio = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        peg_ratio = info.get('pegRatio', None)
        price_to_book = info.get('priceToBook', None)
        profit_margin = info.get('profitMargins', None)
        roe = info.get('returnOnEquity', None)
        debt_to_equity = info.get('debtToEquity', None)
        revenue_growth = info.get('revenueGrowth', None)
        earnings_growth = info.get('earningsGrowth', None)
        market_cap = info.get('marketCap', 0)
        
        # Dividend info
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            dividend_yield *= 100
        
        # Beta (volatility vs market)
        beta = info.get('beta', None)
        
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
        print(f"❌ Error fetching {ticker}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0

def normalize_score(value, min_val, max_val, invert=False):
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
    return 1 - normalized if invert else normalized

# ============================================================================
# DAY TRADER SCORE (1 day - 1 week timeframe)
# ============================================================================
# Focus: Intraday momentum, volume spikes, short-term volatility
# Good for: Quick in-and-out trades, scalping, momentum plays

def calculate_day_trader_score(data):
    """Score for day trading (1d-1w)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. IMMEDIATE MOMENTUM (40%) - Is it moving NOW?
    # Strong weight on 1-day and 1-week changes
    momentum_1d = normalize_score(data['change_1d'], -3, 3)  # -3% to +3% in 1 day
    momentum_1w = normalize_score(data['change_1w'], -8, 8)  # -8% to +8% in 1 week
    scores['momentum'] = (momentum_1d * 0.6) + (momentum_1w * 0.4)
    
    # 2. VOLUME SPIKE (35%) - Are others jumping in?
    # High volume = institutional activity, news catalyst
    volume_score = normalize_score(data['volume_spike_ratio'], 0.8, 3.0)  # 3x avg = max score
    scores['volume'] = volume_score
    
    # 3. INTRADAY VOLATILITY (25%) - Enough movement to profit?
    # Day traders WANT volatility (but not too much)
    vol_score = normalize_score(data['volatility_1w'], 0, 8)  # 4-6% daily = ideal
    # Penalty for extremely high volatility (too risky)
    if data['volatility_1w'] > 10:
        vol_score *= 0.5
    scores['volatility'] = vol_score
    
    # Bonus: Price above 10-day MA (short-term trend confirmation)
    if data['price_vs_ma10'] > 0:
        scores['momentum'] = min(1.0, scores['momentum'] * 1.15)
    
    weights = {'momentum': 0.40, 'volume': 0.35, 'volatility': 0.25}
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': categorize_day_trade(composite, data)
    }

def categorize_day_trade(score, data):
    """Categorize day trade opportunity"""
    if score > 0.75 and data['volume_spike_ratio'] > 2.0:
        return "🚀 HOT MOMENTUM - Volume spike + strong move"
    elif score > 0.65:
        return "📈 TRENDING - Good intraday setup"
    elif score > 0.45:
        return "⚡ VOLATILE - High risk, high reward"
    elif score > 0.30:
        return "📊 CHOPPY - Wait for clearer signal"
    else:
        return "🛑 AVOID - No momentum or volume"

# ============================================================================
# SWING TRADER SCORE (1 week - 3 months)
# ============================================================================
# Focus: Technical patterns, medium-term trends, breakouts
# Good for: Holding 1-6 weeks, riding trends, pattern trading

def calculate_swing_trader_score(data):
    """Score for swing trading (1w-3m)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. TREND MOMENTUM (35%) - Clear direction over weeks
    momentum_raw = (
        data['change_1w'] * 0.3 +
        data['change_1m'] * 0.4 +
        data['change_3m'] * 0.3
    )
    scores['momentum'] = normalize_score(momentum_raw, -15, 15)
    
    # 2. MOVING AVERAGE ALIGNMENT (30%) - Trend confirmation
    ma_score = 0
    if data['price_vs_ma20'] > 0:
        ma_score += 0.35  # Above 20-day
    if data['price_vs_ma50'] > 0:
        ma_score += 0.35  # Above 50-day
    if data['ma_20'] > data['ma_50']:
        ma_score += 0.30  # Golden cross
    scores['trend_alignment'] = ma_score
    
    # 3. VOLUME CONFIRMATION (20%) - Sustained interest
    volume_score = normalize_score(data['volume_spike_ratio'], 0.7, 2.0)
    scores['volume'] = volume_score
    
    # 4. RSI & POSITION (15%) - Not overbought/oversold
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
        position_score = 0.6  # Potential bounce
    else:
        position_score = 0.4  # Overbought
    
    scores['rsi_position'] = (rsi_score + position_score) / 2
    
    weights = {
        'momentum': 0.35,
        'trend_alignment': 0.30,
        'volume': 0.20,
        'rsi_position': 0.15
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': categorize_swing_trade(composite, data)
    }

def categorize_swing_trade(score, data):
    """Categorize swing trade opportunity"""
    if score > 0.75 and data['price_vs_ma50'] > 0:
        return "🎯 STRONG UPTREND - Ride the wave"
    elif score > 0.65:
        return "📈 BULLISH SETUP - Good risk/reward"
    elif score > 0.50:
        return "👀 CONSOLIDATING - Watch for breakout"
    elif score > 0.35:
        return "⚠️ MIXED SIGNALS - Wait for clarity"
    else:
        return "🔻 DOWNTREND - Avoid or short"

# ============================================================================
# POSITION TRADER SCORE (3 months - 1 year)
# ============================================================================
# Focus: Medium-term trends, light fundamentals, technical + some value
# Good for: Holding 3-12 months, trend following with fundamental filter

def calculate_position_trader_score(data):
    """Score for position trading (3m-1y)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. MEDIUM-TERM MOMENTUM (30%) - Sustained trend
    momentum_raw = (
        data['change_3m'] * 0.5 +
        data['change_1y'] * 0.5
    )
    scores['momentum'] = normalize_score(momentum_raw, -20, 30)
    
    # 2. LONG-TERM TREND (25%) - 200-day MA relationship
    if data['price_vs_ma200'] > 5:  # Well above 200-day
        trend_score = 1.0
    elif data['price_vs_ma200'] > 0:
        trend_score = 0.8
    elif data['price_vs_ma200'] > -5:
        trend_score = 0.5
    else:
        trend_score = 0.2
    
    # Bonus for golden cross
    if data['ma_50'] > data['ma_200']:
        trend_score = min(1.0, trend_score * 1.2)
    
    scores['long_term_trend'] = trend_score
    
    # 3. VALUATION BASICS (25%) - Not crazy expensive
    valuation_score = 0.5  # Default neutral
    
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
    
    # 4. STABILITY (20%) - Not too volatile
    vol_score = normalize_score(data['volatility_3m'], 0, 8, invert=True)
    scores['stability'] = vol_score
    
    weights = {
        'momentum': 0.30,
        'long_term_trend': 0.25,
        'valuation': 0.25,
        'stability': 0.20
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': categorize_position_trade(composite, data)
    }

def categorize_position_trade(score, data):
    """Categorize position trade opportunity"""
    if score > 0.75:
        return "💎 STRONG BUY - Trend + value aligned"
    elif score > 0.65:
        return "✅ BUY - Good setup for months ahead"
    elif score > 0.50:
        return "👍 HOLD - Stable, wait and see"
    elif score > 0.35:
        return "⚠️ CAUTION - Weak trend or overvalued"
    else:
        return "❌ SELL/AVOID - Poor outlook"

# ============================================================================
# LONG-TERM INVESTOR SCORE (1+ years)
# ============================================================================
# Focus: Fundamentals, growth, value, quality, dividends
# Good for: Buy and hold, retirement accounts, building wealth

def calculate_longterm_investor_score(data):
    """Score for long-term investing (1y+)"""
    if not data:
        return None
    
    scores = {}
    
    # 1. VALUATION (30%) - Is it reasonably priced?
    valuation_score = 0.5  # Start neutral
    
    if data['pe_ratio'] and data['pe_ratio'] > 0:
        if data['pe_ratio'] < 15:
            valuation_score += 0.25
        elif data['pe_ratio'] < 25:
            valuation_score += 0.10
        elif data['pe_ratio'] > 50:
            valuation_score -= 0.25
    
    if data['peg_ratio'] and data['peg_ratio'] > 0:
        if data['peg_ratio'] < 1.0:
            valuation_score += 0.25  # Growth at reasonable price
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
        if data['profit_margin'] > 0.20:  # >20% margin
            quality_score += 0.25
        elif data['profit_margin'] > 0.10:
            quality_score += 0.10
    
    if data['roe'] and data['roe'] > 0:
        if data['roe'] > 0.15:  # >15% ROE
            quality_score += 0.25
        elif data['roe'] > 0.10:
            quality_score += 0.10
    
    scores['quality'] = max(0, min(1, quality_score))
    
    # 3. GROWTH (25%) - Revenue & earnings growth
    growth_score = 0.5
    
    if data['revenue_growth'] and data['revenue_growth'] > 0:
        if data['revenue_growth'] > 0.15:  # >15% growth
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
        if data['debt_to_equity'] < 50:  # Low debt
            health_score += 0.25
        elif data['debt_to_equity'] > 150:  # High debt
            health_score -= 0.25
    
    if data['dividend_yield'] and data['dividend_yield'] > 0:
        if data['dividend_yield'] > 3:
            health_score += 0.25  # Good income
        elif data['dividend_yield'] > 1.5:
            health_score += 0.15
    
    scores['health_income'] = max(0, min(1, health_score))
    
    weights = {
        'valuation': 0.30,
        'quality': 0.25,
        'growth': 0.25,
        'health_income': 0.20
    }
    
    composite = sum(scores[k] * weights[k] for k in scores.keys())
    
    return {
        'score': composite,
        'components': scores,
        'weights': weights,
        'category': categorize_longterm_investment(composite, data)
    }

def categorize_longterm_investment(score, data):
    """Categorize long-term investment"""
    if score > 0.75:
        return "⭐ EXCELLENT - Strong fundamentals, buy & hold"
    elif score > 0.65:
        return "👍 GOOD - Solid company for long term"
    elif score > 0.50:
        return "✋ FAIR - Okay but not exciting"
    elif score > 0.35:
        return "⚠️ BELOW AVERAGE - Weak fundamentals"
    else:
        return "🚫 POOR - Avoid for long-term portfolio"

# ============================================================================
# MASTER SCORING FUNCTION
# ============================================================================

def calculate_all_scores(ticker):
    """Calculate all timeframe scores for a stock"""
    print(f"\n{'='*80}")
    print(f"📊 MULTI-TIMEFRAME ANALYSIS: {ticker}")
    print(f"{'='*80}")
    
    data = fetch_comprehensive_stock_data(ticker)
    
    if not data:
        print(f"❌ Could not fetch data for {ticker}")
        return None
    
    # Calculate all scores
    day_score = calculate_day_trader_score(data)
    swing_score = calculate_swing_trader_score(data)
    position_score = calculate_position_trader_score(data)
    longterm_score = calculate_longterm_investor_score(data)
    
    # Print summary
    print(f"\n💰 Current Price: ${data['current_price']:.2f}")
    print(f"📈 Market Cap: ${data['market_cap']:,.0f}" if data['market_cap'] else "")
    
    print(f"\n{'TIMEFRAME':<20} {'SCORE':<10} {'RATING':<15} {'RECOMMENDATION'}")
    print("-" * 80)
    
    if day_score:
        print(f"{'Day Trade (1d-1w)':<20} {day_score['score']:.3f}     {'█' * int(day_score['score'] * 10):<15} {day_score['category']}")
    
    if swing_score:
        print(f"{'Swing Trade (1w-3m)':<20} {swing_score['score']:.3f}     {'█' * int(swing_score['score'] * 10):<15} {swing_score['category']}")
    
    if position_score:
        print(f"{'Position (3m-1y)':<20} {position_score['score']:.3f}     {'█' * int(position_score['score'] * 10):<15} {position_score['category']}")
    
    if longterm_score:
        print(f"{'Long-term (1y+)':<20} {longterm_score['score']:.3f}     {'█' * int(longterm_score['score'] * 10):<15} {longterm_score['category']}")
    
    # Detailed breakdown
    print(f"\n📊 DETAILED METRICS:")
    print(f"  Price Changes: 1D: {data['change_1d']:+.2f}% | 1W: {data['change_1w']:+.2f}% | 1M: {data['change_1m']:+.2f}% | 3M: {data['change_3m']:+.2f}% | 1Y: {data['change_1y']:+.2f}%")
    print(f"  Volume: Current: {data['current_volume']:,.0f} | 30D Avg: {data['avg_volume_30d']:,.0f} | Spike Ratio: {data['volume_spike_ratio']:.2f}x")
    print(f"  Volatility: 1W: {data['volatility_1w']:.2f}% | 1M: {data['volatility_1m']:.2f}% | 3M: {data['volatility_3m']:.2f}% | 1Y: {data['volatility_1y']:.2f}%")
    print(f"  Moving Averages: 10D: ${data['ma_10']:.2f} | 20D: ${data['ma_20']:.2f} | 50D: ${data['ma_50']:.2f} | 200D: ${data['ma_200']:.2f}")
    print(f"  RSI: 7-day: {data['rsi_7']:.1f} | 14-day: {data['rsi_14']:.1f}")
    print(f"  52-Week Range: ${data['low_52w']:.2f} - ${data['high_52w']:.2f} (Current: {data['price_position_52w']:.1f}%)")
    
    if data['pe_ratio']:
        print(f"\n📈 FUNDAMENTALS:")
        print(f"  P/E Ratio: {data['pe_ratio']:.2f}" if data['pe_ratio'] else "  P/E Ratio: N/A")
        print(f"  PEG Ratio: {data['peg_ratio']:.2f}" if data['peg_ratio'] else "  PEG Ratio: N/A")
        print(f"  Price/Book: {data['price_to_book']:.2f}" if data['price_to_book'] else "  Price/Book: N/A")
        print(f"  Profit Margin: {data['profit_margin']*100:.2f}%" if data['profit_margin'] else "  Profit Margin: N/A")
        print(f"  ROE: {data['roe']*100:.2f}%" if data['roe'] else "  ROE: N/A")
        print(f"  Dividend Yield: {data['dividend_yield']:.2f}%" if data['dividend_yield'] else "  Dividend Yield: 0%")
    
    print(f"\n{'='*80}")
    
    return {
        'ticker': ticker,
        'data': data,
        'day_trader': day_score,
        'swing_trader': swing_score,
        'position_trader': position_score,
        'longterm_investor': longterm_score
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test with multiple stocks
    test_tickers = ['AAPL', 'TSLA', 'NET', 'NVDA']
    
    results = {}
    for ticker in test_tickers:
        results[ticker] = calculate_all_scores(ticker)
        time.sleep(1)  # Be nice to Yahoo Finance
    
    # Summary comparison
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARISON ACROSS ALL STOCKS")
    print(f"{'='*80}")
    
    print(f"\n{'Ticker':<10} {'Day Trade':<12} {'Swing Trade':<12} {'Position':<12} {'Long-term':<12}")
    print("-" * 80)
    
    for ticker, result in results.items():
        if result:
            day = result['day_trader']['score'] if result['day_trader'] else 0
            swing = result['swing_trader']['score'] if result['swing_trader'] else 0
            position = result['position_trader']['score'] if result['position_trader'] else 0
            longterm = result['longterm_investor']['score'] if result['longterm_investor'] else 0
            
            print(f"{ticker:<10} {day:.3f}        {swing:.3f}        {position:.3f}        {longterm:.3f}")
    
    print(f"\n{'='*80}")
    print("💡 INTERPRETATION GUIDE:")
    print("  - Day Trade: Best for quick momentum plays (minutes to days)")
    print("  - Swing Trade: Best for riding trends (weeks to months)")  
    print("  - Position: Best for medium-term holds (months)")
    print("  - Long-term: Best for buy-and-hold investing (years)")
    print(f"{'='*80}")