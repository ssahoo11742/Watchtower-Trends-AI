import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Commodities:
    def __init__(self, ticker, 
                 produces_excel='produces.xlsx',
                 requires_domestic_excel='requires_domestic.xlsx',
                 requires_total_excel='requires_total.xlsx',
                 naics_mapping_csv='naics_bea_mapping.csv', 
                 companies_csv='companies.csv'):
        """
        Initialize commodity mapper for a company ticker.
        
        Args:
            ticker: Company ticker symbol
            produces_excel: Path to BEA commodity market share Excel file
            requires_domestic_excel: Path to BEA domestic direct requirements Excel file
            requires_total_excel: Path to BEA total (direct + indirect) requirements Excel file
            naics_mapping_csv: Path to CSV with NAICS Code, BEA Industry Code, Industry Title
            companies_csv: Path to companies CSV file
        """
        self.ticker = ticker
        self.companies_csv = companies_csv
        
        # Load BEA commodity production data
        df_produces = pd.read_excel(produces_excel, sheet_name='2017', header=5)
        self.descriptions = dict(zip(df_produces['indCode'], df_produces['indDescr']))
        
        # Load BEA domestic direct requirements data
        df_requires_domestic = pd.read_excel(requires_domestic_excel, sheet_name='2017', header=4)
        self.requirements_descriptions = dict(zip(df_requires_domestic['Code'], 
                                                   df_requires_domestic['Commodity Description']))
        
        # Load BEA total requirements data
        df_requires_total = pd.read_excel(requires_total_excel, sheet_name='2017', header=4)
        
        # Load exact descriptions from CSV mapping
        naics_df = pd.read_csv(naics_mapping_csv)
        self.exact_descriptions = {}
        for _, row in naics_df.iterrows():
            bea_code = row['BEA Industry Code']
            exact_desc = row['Industry Title']
            self.exact_descriptions[bea_code] = exact_desc
        
        # Prepare commodity production data
        df_produces = df_produces.set_index('indCode').drop(columns=['indDescr'])
        self.produces_data = df_produces.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Prepare domestic direct requirements data
        df_requires_domestic = df_requires_domestic.set_index('Code').drop(columns=['Commodity Description'])
        self.requires_domestic_data = df_requires_domestic.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Prepare total requirements data
        df_requires_total = df_requires_total.set_index('Code').drop(columns=['Commodity Description'])
        self.requires_total_data = df_requires_total.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Generate produces and requires attributes
        self.produces = self._generate_produces()
        self.requires = self._generate_requires()
    
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
        if code not in self.produces_data.index:
            code = int(naics_code) if str(naics_code).isdigit() else str(naics_code)
        
        if code not in self.produces_data.index:
            return []
        
        # Get top commodities
        top = self.produces_data.loc[code].sort_values(ascending=False).head(top_n)
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
    
    def get_top_requirements(self, naics_code, top_n=5, include_indirect=True):
        """
        Get the top N requirements for a given NAICS code.
        
        Args:
            naics_code: NAICS code to query
            top_n: Number of top requirements to return (default: 5)
            include_indirect: If True, use total requirements (direct + indirect)
        
        Returns:
            List of dicts with naics_code, requirement_value, description, and layer
        """
        # Try both string and int formats
        code = naics_code
        
        # Get direct requirements
        if code not in self.requires_domestic_data.columns:
            code = int(naics_code) if str(naics_code).isdigit() else str(naics_code)
        
        results = []
        
        # Direct requirements
        if code in self.requires_domestic_data.columns:
            direct = self.requires_domestic_data[code].sort_values(ascending=False).head(top_n)
            direct = direct[direct > 0]
            
            for requirement_code, requirement_value in direct.items():
                results.append({
                    'naics_code': str(requirement_code),
                    'requirement_value': float(requirement_value),
                    'layer': 'direct',
                    'description': self.exact_descriptions.get(requirement_code,
                                  self.requirements_descriptions.get(requirement_code, 'N/A'))
                })
        
        # Indirect requirements (total - direct)
        if include_indirect and code in self.requires_total_data.columns:
            # Calculate indirect by subtracting direct from total
            total_series = self.requires_total_data[code]
            direct_series = self.requires_domestic_data[code] if code in self.requires_domestic_data.columns else pd.Series(0, index=total_series.index)
            
            indirect = total_series - direct_series
            indirect = indirect[indirect > 0].sort_values(ascending=False).head(top_n)
            
            for requirement_code, requirement_value in indirect.items():
                results.append({
                    'naics_code': str(requirement_code),
                    'requirement_value': float(requirement_value),
                    'layer': 'indirect',
                    'description': self.exact_descriptions.get(requirement_code,
                                  self.requirements_descriptions.get(requirement_code, 'N/A'))
                })
        
        return results
    
    def _merge_duplicates(self, items_list):
        """
        Merge duplicate NAICS codes by summing values and averaging confidence.
        
        Args:
            items_list: List of dicts with naics_code, value, and confidence
        
        Returns:
            List of dicts with duplicates merged
        """
        df = pd.DataFrame(items_list)
        if df.empty:
            return []
        
        # Determine value column name
        value_col = 'production_value' if 'production_value' in df.columns else 'requirement_value'
        
        # Group by naics_code and aggregate
        grouped = df.groupby('naics_code').agg({
            value_col: 'sum',
            'confidence': 'mean',
            'description': 'first',  # Keep first description
            **({'layer': 'first'} if 'layer' in df.columns else {})
        }).reset_index()
        
        return grouped.to_dict('records')
    
    def _normalize_values(self, items_list, by_layer=False):
        """
        Normalize production/requirement values to sum to 1.
        
        Args:
            items_list: List of dicts with naics_code and value
            by_layer: If True, normalize separately for each 'layer' (direct/indirect)
        
        Returns:
            List of dicts with normalized values
        """
        if not items_list:
            return []

        # Determine value column name
        value_col = 'production_value' if 'production_value' in items_list[0] else 'requirement_value'

        if by_layer:
            # Group by layer
            from collections import defaultdict
            layer_groups = defaultdict(list)
            for item in items_list:
                layer = item.get('layer', 'combined')
                layer_groups[layer].append(item)
            
            # Normalize within each layer
            for layer, group in layer_groups.items():
                total = sum(i[value_col] for i in group)
                if total > 0:
                    for i in group:
                        i[value_col] /= total
            # Flatten groups
            normalized_items = [i for group in layer_groups.values() for i in group]
            return normalized_items

        else:
            # Normalize all together
            total = sum(item[value_col] for item in items_list)
            if total > 0:
                for item in items_list:
                    item[value_col] /= total
            return items_list

    def _generate_produces(self):
        """
        Generate the produces attribute by combining company NAICS codes 
        with their commodity productions. Merges duplicates and normalizes.
        
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
        
        # Merge duplicates
        all_commodities = self._merge_duplicates(all_commodities)
        
        # Normalize values
        all_commodities = self._normalize_values(all_commodities)
        
        return all_commodities
    
    def _generate_requires(self):
        """
        Generate the requires attribute by combining company NAICS codes 
        with their requirements (direct + indirect). Merges duplicates and normalizes.
        
        Returns:
            list[dict]: List of dicts with naics_code, requirement_value, confidence, and layer
        """
        # Get company's NAICS codes
        company_naics = self.get_company_naics()
        print("NAICS matches:", company_naics)
        all_requirements = []
        
        # For each company NAICS code, get its top requirements
        for naics_match in company_naics:
            parent_code = naics_match['code']
            parent_confidence = naics_match['confidence']
            
            # Get top requirements (direct + indirect)
            requirements = self.get_top_requirements(parent_code, top_n=5, include_indirect=True)
            
            # Add parent confidence to each requirement
            for requirement in requirements:
                all_requirements.append({
                    'naics_code': requirement['naics_code'],
                    'requirement_value': requirement['requirement_value'],
                    'confidence': parent_confidence,
                    'layer': requirement['layer'],
                    'description': requirement['description']
                })
        
        # Merge duplicates (will sum requirement_values from different layers)
        all_requirements = self._merge_duplicates(all_requirements)
        
        # Normalize values
        all_requirements = self._normalize_values(all_requirements, by_layer=True)

        
        return all_requirements


# Example usage
if __name__ == "__main__":
    commodities = Commodities("LAES")
    
    print("PRODUCES (merged & normalized):")
    for item in commodities.produces:
        print(f"  {item['naics_code']}: {item['production_value']:.6f} (conf: {item['confidence']:.3f}) - {item['description']}")
    
    print(f"\nTotal produces sum: {sum(i['production_value'] for i in commodities.produces):.6f}")
    
    print("\nREQUIRES (merged & normalized, direct + indirect):")
    for item in commodities.requires:
        layer = item.get('layer', 'combined')
        print(f"  {item['naics_code']}: {item['requirement_value']:.6f} (conf: {item['confidence']:.3f}) [{layer}] - {item['description']}")
    
    print(f"\nTotal requires sum: {sum(i['requirement_value'] for i in commodities.requires):.6f}")