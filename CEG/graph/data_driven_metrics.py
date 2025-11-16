"""
Data-Driven Edge Metrics Calculator
====================================

Replaces arbitrary magnitude/relevance values with real data:
- Sector: Company's revenue in sector / Total sector market cap
- Industry: Company's market share in industry
- Country: Company GDP / Country GDP
- All: Market correlation, beta similarity, revenue exposure
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataDrivenMetrics:
    """Calculate data-driven edge weights instead of arbitrary values."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.info = yf.Ticker(ticker).info
        self.market_cap = self.info.get('marketCap', 0)
        
        # Cache for expensive calculations
        self._sector_data_cache = None
        self._country_data_cache = None
        
    # ============================================
    # SECTOR METRICS (Data-Driven)
    # ============================================
    
    def get_sector_metrics(self, sector_name: str) -> Dict:
        """
        Calculate data-driven sector metrics.
        
        Returns:
            - magnitude_sector_to_company: Sector's influence on company (beta-based)
            - magnitude_company_to_sector: Company's market share in sector
            - relevance: Classification confidence
        """
        
        # 1. Company → Sector: Market share
        sector_market_cap = self._get_sector_market_cap(sector_name)
        if sector_market_cap > 0:
            company_market_share = self.market_cap / sector_market_cap
            company_to_sector_magnitude = min(0.95, company_market_share * 100)  # Scale to 0-0.95
        else:
            company_to_sector_magnitude = 0.01
        
        # 2. Sector → Company: Beta exposure (how much sector trends affect company)
        sector_beta = self._get_sector_beta(sector_name)
        company_beta = self.info.get('beta', 1.0) or 1.0
        
        # If company beta is close to sector beta, sector influences it strongly
        beta_similarity = 1 - min(1.0, abs(company_beta - sector_beta) / 2.0)
        sector_to_company_magnitude = beta_similarity
        
        # 3. Relevance: Based on revenue concentration
        revenue_concentration = self._get_sector_revenue_concentration(sector_name)
        relevance = 0.70 + (0.25 * revenue_concentration)  # 0.70-0.95 range
        
        return {
            'sector_to_company_magnitude': round(sector_to_company_magnitude, 4),
            'company_to_sector_magnitude': round(company_to_sector_magnitude, 4),
            'relevance': round(relevance, 4),
            'market_share_pct': round(company_market_share * 100, 4) if sector_market_cap > 0 else 0,
            'sector_market_cap': sector_market_cap,
            'company_beta': company_beta,
            'sector_beta': sector_beta,
            'beta_similarity': round(beta_similarity, 4)
        }
    
    def _get_sector_market_cap(self, sector_name: str) -> float:
        """
        Estimate total sector market cap.
        Uses sector ETF as proxy (scaled by typical coverage).
        """
        sector_etf_map = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Basic Materials': 'XLB',
            'Communication Services': 'XLC',
        }
        
        etf = sector_etf_map.get(sector_name)
        if not etf:
            # Fallback: Use company market cap * 100 as rough estimate
            return self.market_cap * 100
        
        try:
            etf_info = yf.Ticker(etf).info
            etf_market_cap = etf_info.get('totalAssets', 0)
            
            # ETFs typically hold 30-50% of sector market cap
            # Scale up by ~2.5x to estimate total sector
            estimated_sector_cap = etf_market_cap * 2.5
            
            return estimated_sector_cap if estimated_sector_cap > 0 else self.market_cap * 100
        except:
            return self.market_cap * 100
    
    def _get_sector_beta(self, sector_name: str) -> float:
        """Get sector beta using sector ETF."""
        sector_etf_map = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Industrials': 'XLI',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Basic Materials': 'XLB',
            'Communication Services': 'XLC',
        }
        
        etf = sector_etf_map.get(sector_name)
        if etf:
            try:
                return yf.Ticker(etf).info.get('beta', 1.0) or 1.0
            except:
                pass
        return 1.0
    
    def _get_sector_revenue_concentration(self, sector_name: str) -> float:
        """
        Estimate what % of company revenue comes from this sector.
        
        Heuristics:
        - If company name contains sector keywords → high concentration
        - If company is diversified (holdings/group) → low concentration
        - Otherwise → medium concentration
        """
        company_name = self.info.get('longName', '').lower()
        
        # Check for diversification signals
        if any(word in company_name for word in ['holdings', 'group', 'conglomerate', 'ventures']):
            return 0.4  # Diversified
        
        # Check for sector-specific keywords
        sector_keywords = {
            'Technology': ['tech', 'software', 'cloud', 'ai', 'data', 'cyber'],
            'Healthcare': ['health', 'pharma', 'medical', 'biotech', 'drug'],
            'Financial Services': ['bank', 'financial', 'insurance', 'capital', 'investment'],
            'Consumer Cyclical': ['retail', 'consumer', 'automotive', 'apparel'],
            'Energy': ['energy', 'oil', 'gas', 'petroleum', 'renewable'],
            'Industrials': ['industrial', 'manufacturing', 'aerospace', 'defense'],
        }
        
        keywords = sector_keywords.get(sector_name, [])
        if any(kw in company_name for kw in keywords):
            return 0.9  # Focused
        
        return 0.6  # Medium concentration (default)
    
    # ============================================
    # INDUSTRY METRICS (Data-Driven)
    # ============================================
    
    def get_industry_metrics(self, industry_name: str) -> Dict:
        """
        Calculate data-driven industry metrics.
        Industries are more specific than sectors, so magnitudes are generally higher.
        """
        
        # Similar to sector but with tighter coupling
        # Industry market cap is smaller, so company can have higher share
        industry_market_cap = self._get_industry_market_cap(industry_name)
        
        if industry_market_cap > 0:
            market_share = self.market_cap / industry_market_cap
            company_to_industry_magnitude = min(0.98, market_share * 50)  # More influence in smaller industry
        else:
            company_to_industry_magnitude = 0.02
        
        # Industry → Company: Even stronger correlation than sector
        industry_beta = self._get_industry_beta(industry_name)
        company_beta = self.info.get('beta', 1.0) or 1.0
        
        beta_similarity = 1 - min(1.0, abs(company_beta - industry_beta) / 1.5)
        industry_to_company_magnitude = min(0.98, beta_similarity * 1.1)  # Slightly higher than sector
        
        # Relevance: Higher for industry (more specific classification)
        revenue_concentration = self._get_industry_revenue_concentration(industry_name)
        relevance = 0.75 + (0.20 * revenue_concentration)  # 0.75-0.95 range
        
        return {
            'industry_to_company_magnitude': round(industry_to_company_magnitude, 4),
            'company_to_industry_magnitude': round(company_to_industry_magnitude, 4),
            'relevance': round(relevance, 4),
            'market_share_pct': round(market_share * 100, 4) if industry_market_cap > 0 else 0,
            'industry_market_cap': industry_market_cap,
            'company_beta': company_beta,
            'industry_beta': industry_beta
        }
    
    def _get_industry_market_cap(self, industry_name: str) -> float:
        """Estimate industry market cap (typically 1/10 of sector)."""
        sector = self.info.get('sector', 'Unknown')
        sector_cap = self._get_sector_market_cap(sector)
        
        # Industries are typically 5-15% of their sector
        return sector_cap * 0.10
    
    def _get_industry_beta(self, industry_name: str) -> float:
        """Get industry beta (use company beta as proxy for now)."""
        return self.info.get('beta', 1.0) or 1.0
    
    def _get_industry_revenue_concentration(self, industry_name: str) -> float:
        """Industry revenue concentration (higher than sector)."""
        sector_conc = self._get_sector_revenue_concentration(self.info.get('sector', ''))
        return min(0.95, sector_conc * 1.2)  # Industry is more specific
    
    # ============================================
    # COUNTRY METRICS (Data-Driven)
    # ============================================
    
    def get_country_metrics(self, country_name: str) -> Dict:
        """
        Calculate data-driven country metrics.
        
        Uses:
        - Country GDP data
        - Company revenue/market cap
        - Regulatory exposure
        """
        
        country_gdp = self._get_country_gdp(country_name)
        company_revenue = self.info.get('totalRevenue', self.market_cap * 0.15)  # Fallback to ~15% of market cap
        
        # Company → Country: Economic footprint (revenue / GDP)
        if country_gdp > 0:
            gdp_share = company_revenue / country_gdp
            company_to_country_magnitude = min(0.95, gdp_share * 1000)  # Scale up small percentages
        else:
            company_to_country_magnitude = 0.01
        
        # Country → Company: Regulatory/market influence
        # Larger economies have more regulatory power
        country_regulatory_strength = self._get_country_regulatory_strength(country_name)
        
        # Also consider: what % of company revenue is from this country
        revenue_exposure = self._get_country_revenue_exposure(country_name)
        
        country_to_company_magnitude = (0.4 * country_regulatory_strength + 
                                       0.6 * revenue_exposure)
        
        # Relevance: Always 1.0 for country (headquarters is certain)
        relevance = 1.0
        
        return {
            'country_to_company_magnitude': round(country_to_company_magnitude, 4),
            'company_to_country_magnitude': round(company_to_country_magnitude, 4),
            'relevance': relevance,
            'gdp_share_pct': round(gdp_share * 100, 8) if country_gdp > 0 else 0,
            'country_gdp': country_gdp,
            'company_revenue': company_revenue,
            'revenue_exposure': round(revenue_exposure, 4),
            'regulatory_strength': round(country_regulatory_strength, 4)
        }
    
    def _get_country_gdp(self, country_name: str) -> float:
        """
        Get country GDP (2024 estimates in USD).
        Source: World Bank / IMF data
        """
        gdp_data = {
            'United States': 27.36e12,
            'China': 17.96e12,
            'Japan': 4.41e12,
            'Germany': 4.31e12,
            'India': 3.94e12,
            'United Kingdom': 3.34e12,
            'France': 3.05e12,
            'Italy': 2.25e12,
            'Canada': 2.14e12,
            'Brazil': 2.13e12,
            'South Korea': 1.71e12,
            'Australia': 1.69e12,
            'Spain': 1.58e12,
            'Mexico': 1.57e12,
            'Indonesia': 1.42e12,
            'Netherlands': 1.09e12,
            'Saudi Arabia': 1.06e12,
            'Turkey': 1.03e12,
            'Switzerland': 0.91e12,
            'Taiwan': 0.79e12,
            'Poland': 0.84e12,
            'Belgium': 0.63e12,
            'Sweden': 0.62e12,
            'Ireland': 0.59e12,
            'Israel': 0.54e12,
            'Singapore': 0.52e12,
            'Vietnam': 0.43e12,
        }
        
        return gdp_data.get(country_name, 1e12)  # Default to $1T if unknown
    
    def _get_country_regulatory_strength(self, country_name: str) -> float:
        """
        Regulatory strength: How much country can influence company operations.
        
        Factors:
        - Rule of law
        - Economic size
        - Market openness
        """
        # Simplified scoring (0.3-1.0)
        regulatory_scores = {
            'United States': 0.95,
            'China': 0.90,
            'European Union': 0.95,
            'Germany': 0.90,
            'United Kingdom': 0.88,
            'France': 0.87,
            'Japan': 0.85,
            'South Korea': 0.82,
            'Canada': 0.85,
            'Australia': 0.83,
            'Singapore': 0.88,
            'India': 0.75,
            'Brazil': 0.65,
            'Mexico': 0.60,
            'Vietnam': 0.55,
        }
        
        return regulatory_scores.get(country_name, 0.70)
    
    def _get_country_revenue_exposure(self, country_name: str) -> float:
        """
        Estimate what % of revenue comes from this country.
        
        Heuristics:
        - If country = headquarters → assume 40-70% domestic revenue
        - Adjust based on company size (larger = more international)
        """
        
        # Large companies tend to be more international
        if self.market_cap > 100e9:  # >$100B
            domestic_revenue_share = 0.45
        elif self.market_cap > 10e9:  # >$10B
            domestic_revenue_share = 0.60
        else:
            domestic_revenue_share = 0.75
        
        # Check if this is headquarters country
        company_country = self.info.get('country', 'Unknown')
        if company_country == country_name:
            return domestic_revenue_share
        else:
            # Foreign market - estimate based on trade relationships
            return 0.15  # Assume 15% from other countries on average
    
    # ============================================
    # CORRELATION STRENGTH (Market-Based)
    # ============================================
    
    def get_correlation_metrics(self, edge_type: str, proxy_ticker: Optional[str] = None) -> Dict:
        """
        Calculate market-based correlation strength.
        
        This is separate from structural magnitude.
        """
        if not proxy_ticker:
            return {
                'price_correlation': 0.0,
                'beta_similarity': 0.0,
                'correlation_strength': 0.0
            }
        
        try:
            # Price correlation
            price_corr = self._compute_price_correlation(self.ticker, proxy_ticker)
            
            # Beta similarity
            beta1 = self.info.get('beta', 1.0) or 1.0
            beta2 = yf.Ticker(proxy_ticker).info.get('beta', 1.0) or 1.0
            beta_sim = 1 / (1 + abs(beta1 - beta2))
            
            # Combined correlation strength
            correlation_strength = 0.7 * price_corr + 0.3 * beta_sim
            
            return {
                'price_correlation': round(price_corr, 4),
                'beta_similarity': round(beta_sim, 4),
                'correlation_strength': round(correlation_strength, 4),
                'beta1': beta1,
                'beta2': beta2
            }
        except Exception as e:
            print(f"  ⚠️  Correlation calculation failed: {e}")
            return {
                'price_correlation': 0.0,
                'beta_similarity': 0.0,
                'correlation_strength': 0.0
            }
    
    def _compute_price_correlation(self, ticker1: str, ticker2: str, period: str = "1y") -> float:
        """Compute price correlation between two tickers."""
        try:
            df1 = yf.download(ticker1, period=period, progress=False)
            df2 = yf.download(ticker2, period=period, progress=False)
            
            if df1.empty or df2.empty:
                return 0.0
            
            # Extract close prices
            close1 = df1['Close'].squeeze()
            close2 = df2['Close'].squeeze()
            
            # Align and calculate returns
            df = pd.DataFrame({'t1': close1, 't2': close2}).dropna()
            
            if len(df) < 20:
                return 0.0
            
            returns1 = df['t1'].pct_change().dropna()
            returns2 = df['t2'].pct_change().dropna()
            
            if len(returns1) < 10:
                return 0.0
            
            corr = float(np.corrcoef(returns1, returns2)[0, 1])
            
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Example: Calculate metrics for Apple
    metrics = DataDrivenMetrics("AAPL")
    
    print("=" * 60)
    print("SECTOR METRICS (Technology)")
    print("=" * 60)
    sector_metrics = metrics.get_sector_metrics("Technology")
    for key, value in sector_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("INDUSTRY METRICS (Consumer Electronics)")
    print("=" * 60)
    industry_metrics = metrics.get_industry_metrics("Consumer Electronics")
    for key, value in industry_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("COUNTRY METRICS (United States)")
    print("=" * 60)
    country_metrics = metrics.get_country_metrics("United States")
    for key, value in country_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("CORRELATION METRICS (vs XLK)")
    print("=" * 60)
    corr_metrics = metrics.get_correlation_metrics("sector", "XLK")
    for key, value in corr_metrics.items():
        print(f"  {key}: {value}")