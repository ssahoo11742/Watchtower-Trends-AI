import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Commodities:
    def __init__(self, ticker, excel_file='bea_industry_classification.xlsx', 
                 naics_mapping_csv='naics_bea_mapping.csv', 
                 companies_csv='companies.csv'):
        """
        Initialize commodity mapper for a company ticker.
        
        Args:
            ticker: Company ticker symbol
            excel_file: Path to BEA commodity market share Excel file
            naics_mapping_csv: Path to CSV with NAICS Code, BEA Industry Code, Industry Title
            companies_csv: Path to companies CSV file
        """
        self.ticker = ticker
        self.companies_csv = companies_csv
        
        # Load BEA commodity data
        df = pd.read_excel(excel_file, sheet_name='2017', header=5)
        self.descriptions = dict(zip(df['indCode'], df['indDescr']))
        
        # Load exact descriptions from CSV mapping
        naics_df = pd.read_csv(naics_mapping_csv)
        self.exact_descriptions = {}
        for _, row in naics_df.iterrows():
            bea_code = row['BEA Industry Code']
            exact_desc = row['Industry Title']
            self.exact_descriptions[bea_code] = exact_desc
        
        # Prepare commodity data
        df = df.set_index('indCode').drop(columns=['indDescr'])
        self.data = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Generate produces attribute
        self.produces = self._generate_produces()
    
    def get_company_naics(self, min_confidence=0.3):
        """
        Given a company ticker, match its business description to likely NAICS codes.
        
        Args:
            min_confidence (float): Minimum cosine similarity to include a NAICS match.
        
        Returns:
            list[dict]: Top NAICS matches with 'code', 'label', and 'confidence'.
        """
        # Load company data
        companies = pd.read_csv(self.companies_csv)
        
        # Check for ticker
        if self.ticker not in companies["Ticker"].values:
            raise ValueError(f"Ticker '{self.ticker}' not found in companies.csv.")

        company = companies[companies["Ticker"] == self.ticker].iloc[0]
        description = company["Description"]
        
        # Get NAICS dataset from Census API
        url = "https://api.census.gov/data/2022/cbp?get=NAICS2017,NAICS2017_LABEL,ESTAB,EMP&for=state:*"
        data = requests.get(url).json()
        
        columns = data[0]
        rows = data[1:]
        naics_df = pd.DataFrame(rows, columns=columns)
        
        # Drop duplicates and nulls
        naics_df = naics_df.drop_duplicates(subset=["NAICS2017_LABEL"]).dropna(subset=["NAICS2017_LABEL"])
        
        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(naics_df["NAICS2017_LABEL"].astype(str))
        
        # Compute similarity
        query_vec = vectorizer.transform([description])
        similarities = cosine_similarity(query_vec, tfidf).flatten()
        
        # Attach similarity scores
        naics_df["confidence"] = similarities
        
        # Filter by confidence threshold
        filtered = naics_df[naics_df["confidence"] > min_confidence]
        
        if filtered.empty:
            # Fallback: top 5 by similarity
            filtered = naics_df.sort_values(by="confidence", ascending=False).head(5)
        else:
            filtered = filtered.sort_values(by="confidence", ascending=False).head(10)
        
        # Format the output
        top_matches = [
            {
                "code": row["NAICS2017"],
                "label": row["NAICS2017_LABEL"],
                "confidence": round(float(row["confidence"]), 3),
            }
            for _, row in filtered.iterrows()
        ]
        
        return top_matches
    
    def get_top_commodities(self, naics_code, top_n=5):
        """
        Get the top N commodities produced by a given NAICS code.
        
        Args:
            naics_code: NAICS code to query
            top_n: Number of top commodities to return (default: 5)
        
        Returns:
            List of dicts with naics_code, production_value, and description
        """
        # Try both string and int formats
        code = naics_code
        if code not in self.data.index:
            code = int(naics_code) if str(naics_code).isdigit() else str(naics_code)
        
        if code not in self.data.index:
            return []
        
        # Get top commodities
        top = self.data.loc[code].sort_values(ascending=False).head(top_n)
        top = top[top > 0]  # Filter out zeros
        
        # Build result as list of dicts
        results = []
        for commodity_code, production_value in top.items():
            results.append({
                'naics_code': str(commodity_code),
                'production_value': float(production_value),
                'description': self.exact_descriptions.get(commodity_code, 
                              self.descriptions.get(commodity_code, 'N/A'))
            })
        
        return results
    
    def _generate_produces(self):
        """
        Generate the produces attribute by combining company NAICS codes 
        with their commodity productions.
        
        Returns:
            list[dict]: List of dicts with naics_code, production_value, and confidence
        """
        # Get company's NAICS codes
        company_naics = self.get_company_naics()
        
        all_commodities = []
        
        # For each company NAICS code, get its top commodities
        for naics_match in company_naics:
            parent_code = naics_match['code']
            parent_confidence = naics_match['confidence']
            
            # Get top 5 commodities for this NAICS code
            commodities = self.get_top_commodities(parent_code, top_n=5)
            
            # Add parent confidence to each commodity
            for commodity in commodities:
                all_commodities.append({
                    'naics_code': commodity['naics_code'],
                    'production_value': commodity['production_value'],
                    'confidence': parent_confidence,
                    'description': commodity['description']
                })
        
        return all_commodities


# Example usage
if __name__ == "__main__":
    commodities = Commodities("QS")
    print(commodities.produces)