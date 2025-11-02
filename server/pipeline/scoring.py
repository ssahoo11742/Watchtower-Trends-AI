import yfinance as yf


# ============================================================================
# COMPREHENSIVE STOCK DATA FETCHING
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
