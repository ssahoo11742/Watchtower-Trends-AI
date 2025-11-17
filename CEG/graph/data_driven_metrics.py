"""
Robust Data-Driven Edge Metrics Calculator with Sector Market Cap
=================================================================

Uses companies_cap_by_sector.csv for accurate market cap calculations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ETFMatcher:
    """Matches sectors/industries/commodities to ETFs using keyword similarity."""
    
    def __init__(self, companies_csv: str = "./data/companies.csv"):
        """Load ETF data from CSV."""
        try:
            df = pd.read_csv(companies_csv)
            # Filter for ETFs only
            self.etfs = df[df['Name'].str.contains("ETF", case=False, na=False)].copy()
            
            # Normalize keywords to list
            self.etfs["KeywordList"] = (
                self.etfs["Keywords"]
                .fillna("")
                .apply(lambda k: [x.strip().lower() for x in str(k).split(",")])
            )
            
            print(f"✅ Loaded {len(self.etfs)} ETFs from {companies_csv}")
        except Exception as e:
            print(f"⚠️  Failed to load ETFs: {e}")
            self.etfs = pd.DataFrame()
    
    def find_etf(self, category: str, name: str) -> Tuple[Optional[str], int]:
        """
        Find best matching ETF for a sector/industry/commodity/country.
        
        Returns:
            (ticker, confidence_score) where score is 0-10
        """
        if self.etfs.empty or not name:
            return None, 0
        
        target = name.lower()
        best_match = None
        best_score = 0
        
        for _, row in self.etfs.iterrows():
            score = 0
            
            # Keyword matching (most important)
            for kw in row["KeywordList"]:
                if not kw:
                    continue
                # Full keyword in target
                if kw in target:
                    score += 3
                # Target in keyword
                if target in kw:
                    score += 3
                # Prefix match (4+ chars)
                if len(target) >= 4 and len(kw) >= 4 and kw.startswith(target[:4]):
                    score += 2
                if len(target) >= 4 and len(kw) >= 4 and target.startswith(kw[:4]):
                    score += 2
            
            # ETF Name matching
            etf_name = row["Name"].lower()
            if target in etf_name:
                score += 4
            if name.lower() in etf_name:
                score += 2
            
            # Category-specific boosts
            if category == "commodity" and any(word in etf_name for word in ["commodity", "materials", "metal", "energy"]):
                score += 1
            if category == "sector" and "sector" in etf_name:
                score += 1
            if category == "country" and any(word in etf_name for word in ["country", "international", "emerging"]):
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = row["Ticker"]
        
        return best_match, best_score


class SectorCapCalculator:
    """Calculate sector and industry total market caps from companies data."""
    
    def __init__(self, cap_csv: str = "companies_cap_by_sector.csv"):
        """Load market cap data."""
        try:
            self.df = pd.read_csv(cap_csv)
            print(f"✅ Loaded {len(self.df)} companies from {cap_csv}")
        except Exception as e:
            print(f"⚠️  Failed to load market cap data: {e}")
            self.df = pd.DataFrame()
    
    def get_sector_market_cap(self, sector_name: str) -> float:
        """Get total market cap for a sector."""
        if self.df.empty:
            return 0
        
        sector_companies = self.df[self.df['Sector'] == sector_name]
        total_cap = sector_companies['MarketCap'].sum()
        return float(total_cap)
    
    def get_industry_market_cap(self, industry_name: str) -> float:
        """Get total market cap for an industry."""
        if self.df.empty:
            return 0
        
        industry_companies = self.df[self.df['Industry'] == industry_name]
        total_cap = industry_companies['MarketCap'].sum()
        return float(total_cap)


class DataDrivenMetrics:
    """
    Calculate proper data-driven edge weights using market cap data.
    """
    
    def __init__(self, ticker: str, companies_csv: str = "./data/companies.csv", cap_csv: str = "companies_cap_by_sector.csv"):
        self.ticker = ticker
        self.info = yf.Ticker(ticker).info
        self.market_cap = self.info.get('marketCap', 0)
        
        # Initialize calculators
        self.etf_matcher = ETFMatcher(companies_csv)
        self.cap_calculator = SectorCapCalculator(cap_csv)
        
        # Cache
        self._cache = {}
    
    def _get_etf_market_cap(self, etf_ticker: str) -> float:
        """Get ETF market cap (total assets)."""
        if not etf_ticker:
            return 0
        
        try:
            etf_info = yf.Ticker(etf_ticker).info
            # ETFs store market cap as "totalAssets"
            return etf_info.get('totalAssets', etf_info.get('marketCap', 0))
        except:
            return 0
    
    def _compute_price_correlation(self, ticker1: str, ticker2: str, period: str = "1y") -> float:
        """Compute price correlation between two tickers."""
        try:
            df1 = yf.download(ticker1, period=period, progress=False)
            df2 = yf.download(ticker2, period=period, progress=False)
            
            if df1.empty or df2.empty:
                return 0.0
            
            # Extract close prices
            def extract_close(df):
                try:
                    return df.xs('Close', level=1, axis=1).squeeze()
                except:
                    try:
                        return df.xs('Close', level=0, axis=1).squeeze()
                    except:
                        return df['Close'].squeeze()
            
            close1 = extract_close(df1)
            close2 = extract_close(df2)
            
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
    # SECTOR METRICS (Market Cap Based)
    # ============================================
    
    def get_sector_metrics(self, sector_name: str) -> Dict:
        """
        Calculate sector metrics using actual sector market cap data.
        """
        
        # Get sector market cap from data
        sector_market_cap = self.cap_calculator.get_sector_market_cap(sector_name)
        
        if sector_market_cap == 0:
            print(f"  ⚠️  No market cap data for sector '{sector_name}'")
            return self._get_fallback_sector_metrics(sector_name)
        
        print(f"  📊 Sector '{sector_name}' market cap: ${sector_market_cap:,.0f}")
        
        # Find sector ETF for correlation
        sector_etf, etf_score = self.etf_matcher.find_etf("sector", sector_name)
        
        if sector_etf:
            print(f"  📊 Matched sector '{sector_name}' → ETF {sector_etf} (score: {etf_score})")
        
        # ============================================
        # COMPANY → SECTOR (BELONGS_TO)
        # ============================================
        
        # Weight = Market share (capped at 95%)
        if self.market_cap > 0:
            market_share = self.market_cap / sector_market_cap
            company_to_sector_weight = min(0.95, market_share)
        else:
            company_to_sector_weight = 0.0
        
        # Confidence = Classification certainty
        classification_confidence = self._get_classification_confidence(sector_name)
        
        # ============================================
        # SECTOR → COMPANY (INFLUENCES)
        # ============================================
        
        # Weight = Price correlation with sector ETF
        if sector_etf:
            price_corr = self._compute_price_correlation(self.ticker, sector_etf)
            sector_to_company_weight = abs(price_corr)
        else:
            sector_to_company_weight = 0.0
        
        # If correlation fails, fall back to beta similarity
        if sector_to_company_weight < 0.1:
            company_beta = self.info.get('beta', 1.0) or 1.0
            if sector_etf:
                etf_beta = yf.Ticker(sector_etf).info.get('beta', 1.0) or 1.0
                beta_similarity = 1 - min(1.0, abs(company_beta - etf_beta) / 2.0)
                sector_to_company_weight = max(sector_to_company_weight, beta_similarity * 0.8)
            else:
                sector_to_company_weight = 0.70
        
        # Cap at 95%
        sector_to_company_weight = min(0.95, sector_to_company_weight)
        
        return {
            # Company → Sector (BELONGS_TO)
            'company_to_sector_weight': round(company_to_sector_weight, 6),
            'company_to_sector_confidence': round(classification_confidence, 4),
            
            # Sector → Company (INFLUENCES)
            'sector_to_company_weight': round(sector_to_company_weight, 4),
            'sector_to_company_confidence': round(classification_confidence, 4),
            
            # Metadata
            'correlation_strength': round(abs(sector_to_company_weight), 4),
            'market_share_pct': round(company_to_sector_weight * 100, 6),
            'sector_market_cap': sector_market_cap,
            'company_market_cap': self.market_cap,
            'sector_etf': sector_etf,
            'etf_match_score': etf_score
        }
    
    def _get_fallback_sector_metrics(self, sector_name: str) -> Dict:
        """Fallback when no market cap data found."""
        # Use rough estimate
        estimated_sector_cap = self.market_cap * 200 if self.market_cap > 0 else 1e12
        market_share = self.market_cap / estimated_sector_cap if estimated_sector_cap > 0 else 0.0
        
        return {
            'company_to_sector_weight': round(min(0.5, market_share), 6),
            'company_to_sector_confidence': 0.60,
            'sector_to_company_weight': 0.70,
            'sector_to_company_confidence': 0.60,
            'correlation_strength': 0.50,
            'market_share_pct': round(market_share * 100, 6),
            'sector_market_cap': estimated_sector_cap,
            'company_market_cap': self.market_cap,
            'sector_etf': None,
            'etf_match_score': 0
        }
    
    # ============================================
    # INDUSTRY METRICS (Market Cap Based)
    # ============================================
    
    def get_industry_metrics(self, industry_name: str) -> Dict:
        """
        Calculate industry metrics using actual industry market cap data.
        """
        
        # Get industry market cap from data
        industry_market_cap = self.cap_calculator.get_industry_market_cap(industry_name)
        
        if industry_market_cap == 0:
            print(f"  ⚠️  No market cap data for industry '{industry_name}', estimating from sector")
            industry_market_cap = self._estimate_industry_cap_from_sector(industry_name)
        else:
            print(f"  🏭 Industry '{industry_name}' market cap: ${industry_market_cap:,.0f}")
        
        # Try to find industry-specific ETF
        industry_etf, etf_score = self.etf_matcher.find_etf("industry", industry_name)
        
        if industry_etf and etf_score >= 5:
            print(f"  🏭 Matched industry '{industry_name}' → ETF {industry_etf} (score: {etf_score})")
            price_corr = self._compute_price_correlation(self.ticker, industry_etf)
            industry_to_company_weight = min(0.90, abs(price_corr) * 1.1)
        else:
            # Use sector correlation as proxy
            sector_name = self.info.get('sector', 'Unknown')
            sector_etf, _ = self.etf_matcher.find_etf("sector", sector_name)
            if sector_etf:
                price_corr = self._compute_price_correlation(self.ticker, sector_etf)
                industry_to_company_weight = min(0.90, abs(price_corr) * 1.15)
            else:
                industry_to_company_weight = 0.75
        
        # Company → Industry weight
        if industry_market_cap > 0 and self.market_cap > 0:
            market_share = self.market_cap / industry_market_cap
            company_to_industry_weight = min(0.95, market_share)
        else:
            company_to_industry_weight = 0.0
        
        # Confidence
        classification_confidence = min(0.95, self._get_classification_confidence(self.info.get('sector', '')) + 0.05)
        
        return {
            'company_to_industry_weight': round(company_to_industry_weight, 6),
            'company_to_industry_confidence': round(classification_confidence, 4),
            'industry_to_company_weight': round(industry_to_company_weight, 4),
            'industry_to_company_confidence': round(classification_confidence, 4),
            'correlation_strength': round(industry_to_company_weight, 4),
            'market_share_pct': round(company_to_industry_weight * 100, 6),
            'industry_market_cap': industry_market_cap,
            'company_market_cap': self.market_cap,
            'industry_etf': industry_etf,
            'etf_match_score': etf_score
        }
    
    def _estimate_industry_cap_from_sector(self, industry_name: str) -> float:
        """Estimate industry cap from sector data."""
        sector_name = self.info.get('sector', 'Unknown')
        sector_market_cap = self.cap_calculator.get_sector_market_cap(sector_name)
        
        if sector_market_cap == 0:
            sector_market_cap = self.market_cap * 200 if self.market_cap > 0 else 1e12
        
        # Estimate industry as 10-25% of sector based on specificity
        industry_lower = industry_name.lower()
        
        # Major/broad industries
        if any(word in industry_lower for word in ['auto', 'bank', 'software', 'pharma', 'oil', 'semiconductor']):
            fraction = 0.25
        else:
            fraction = 0.12
        
        estimated_industry_cap = sector_market_cap * fraction
        
        # Safety: industry must be at least 3x company size
        min_cap = self.market_cap * 3.0
        
        return max(estimated_industry_cap, min_cap)
    
    # ============================================
    # COUNTRY METRICS (ETF-Based)
    # ============================================
    
    def get_country_metrics(self, country_name: str) -> Dict:
        """Calculate country metrics using country ETFs and GDP data."""
        
        # Find country ETF
        country_etf, etf_score = self.etf_matcher.find_etf("country", country_name)
        
        # Get GDP
        country_gdp = self._get_country_gdp(country_name)
        company_revenue = self.info.get('totalRevenue', self.market_cap * 0.15)
        
        # ============================================
        # COMPANY → COUNTRY (LOCATED_IN)
        # ============================================
        
        # Weight = Economic footprint (revenue as % of GDP)
        if country_gdp > 0 and company_revenue > 0:
            gdp_share = company_revenue / country_gdp
            company_to_country_weight = gdp_share
        else:
            company_to_country_weight = 0.0
        
        # ============================================
        # COUNTRY → COMPANY (INFLUENCES)
        # ============================================
        
        # Use country ETF for correlation if available
        if country_etf and etf_score >= 4:
            print(f"  🌍 Matched country '{country_name}' → ETF {country_etf} (score: {etf_score})")
            price_corr = self._compute_price_correlation(self.ticker, country_etf)
            country_influence = abs(price_corr)
        else:
            # Fallback: regulatory strength × revenue exposure
            regulatory_strength = self._get_country_regulatory_strength(country_name)
            revenue_exposure = self._get_country_revenue_exposure(country_name)
            country_influence = regulatory_strength * revenue_exposure
        
        country_to_company_weight = min(0.95, country_influence)
        
        return {
            'company_to_country_weight': round(company_to_country_weight, 8),
            'company_to_country_confidence': 0.98,
            'country_to_company_weight': round(country_to_company_weight, 4),
            'country_to_company_confidence': 0.95,
            'correlation_strength': round(country_to_company_weight, 4),
            'gdp_share_pct': round(company_to_country_weight * 100, 8),
            'country_gdp': country_gdp,
            'company_revenue': company_revenue,
            'country_etf': country_etf,
            'etf_match_score': etf_score
        }
    
    def _get_country_gdp(self, country_name: str) -> float:
        """Get country GDP (2024 estimates)."""
        gdp_data = {
            'United States': 27.36e12, 'China': 17.96e12, 'Japan': 4.41e12,
            'Germany': 4.31e12, 'India': 3.94e12, 'United Kingdom': 3.34e12,
            'France': 3.05e12, 'Italy': 2.25e12, 'Canada': 2.14e12,
            'Brazil': 2.13e12, 'South Korea': 1.71e12, 'Australia': 1.69e12,
            'Spain': 1.58e12, 'Mexico': 1.57e12, 'Indonesia': 1.42e12,
            'Netherlands': 1.09e12, 'Saudi Arabia': 1.06e12, 'Turkey': 1.03e12,
            'Switzerland': 0.91e12, 'Taiwan': 0.79e12, 'Poland': 0.84e12,
            'Belgium': 0.63e12, 'Sweden': 0.62e12, 'Ireland': 0.59e12,
            'Israel': 0.54e12, 'Singapore': 0.52e12, 'Vietnam': 0.43e12,
        }
        return gdp_data.get(country_name, 1e12)
    
    def _get_country_regulatory_strength(self, country_name: str) -> float:
        """Regulatory influence strength."""
        scores = {
            'United States': 0.95, 'China': 0.90, 'Germany': 0.90,
            'United Kingdom': 0.88, 'France': 0.87, 'Japan': 0.85,
            'South Korea': 0.82, 'Canada': 0.85, 'Australia': 0.83,
            'Singapore': 0.88, 'India': 0.75, 'Brazil': 0.65,
        }
        return scores.get(country_name, 0.70)
    
    def _get_country_revenue_exposure(self, country_name: str) -> float:
        """Estimate revenue exposure to country."""
        company_country = self.info.get('country', 'Unknown')
        
        if company_country == country_name:
            # Domestic market
            if self.market_cap > 100e9:
                return 0.45
            elif self.market_cap > 10e9:
                return 0.60
            else:
                return 0.75
        else:
            return 0.10
    
    def _get_classification_confidence(self, sector_name: str) -> float:
        """Classification confidence based on company characteristics."""
        company_name = self.info.get('longName', '').lower()
        
        if any(word in company_name for word in ['holdings', 'group', 'conglomerate']):
            return 0.70
        
        sector_keywords = {
            'Technology': ['tech', 'software', 'cloud', 'ai', 'data'],
            'Healthcare': ['health', 'pharma', 'medical', 'biotech'],
            'Financial Services': ['bank', 'financial', 'insurance'],
        }
        
        keywords = sector_keywords.get(sector_name, [])
        if any(kw in company_name for kw in keywords):
            return 0.95
        
        return 0.85


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("=" * 80)
    print("SECTOR MARKET CAP BASED METRICS")
    print("=" * 80)
    
    metrics = DataDrivenMetrics("QS", companies_csv="./data/companies.csv", cap_csv="./data/companies_cap_by_sector.csv")
    
    print("\n📊 SECTOR METRICS:")
    sector = metrics.get_sector_metrics("Consumer Cyclical")
    print(f"  Company → Sector weight: {sector['company_to_sector_weight']:.6f} ({sector['market_share_pct']:.4f}%)")
    print(f"  Sector → Company weight: {sector['sector_to_company_weight']:.4f}")
    print(f"  ETF: {sector.get('sector_etf')} (score: {sector.get('etf_match_score')})")
    
    print("\n🏭 INDUSTRY METRICS:")
    industry = metrics.get_industry_metrics("Auto Parts")
    print(f"  Company → Industry weight: {industry['company_to_industry_weight']:.6f} ({industry['market_share_pct']:.4f}%)")
    print(f"  Industry → Company weight: {industry['industry_to_company_weight']:.4f}")
    print(f"  ETF: {industry.get('industry_etf')} (score: {industry.get('etf_match_score')})")
    
    print("\n🌍 COUNTRY METRICS:")
    country = metrics.get_country_metrics("United States")
    print(f"  Company → Country weight: {country['company_to_country_weight']:.8f}")
    print(f"  Country → Company weight: {country['country_to_company_weight']:.4f}")
    print(f"  ETF: {country.get('country_etf')} (score: {country.get('etf_match_score')})")