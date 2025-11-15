import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Commodities:
    def __init__(self, ticker, 
                 produces_excel='./data/produces.xlsx',
                 requires_domestic_excel='./data/requires_domestic.xlsx',
                 requires_total_excel='./data/requires_total.xlsx',
                 naics_mapping_csv='./data/naics_bea_mapping.csv', 
                 companies_csv='./data/companies.csv',
                 naics_descriptions_xlsx='./data/2022_NAICS_Descriptions.xlsx',
                 naics_index_xlsx='./data/2022_NAICS_Index_File.xlsx',
                 naics_cross_ref_xlsx='./data/2022_NAICS_Cross_References.xlsx'):
        """
        Initialize commodity mapper for a company ticker.
        
        Args:
            ticker: Company ticker symbol
            produces_excel: Path to BEA commodity market share Excel file
            requires_domestic_excel: Path to BEA domestic direct requirements Excel file
            requires_total_excel: Path to BEA total (direct + indirect) requirements Excel file
            naics_mapping_csv: Path to CSV with NAICS Code, BEA Industry Code, Industry Title
            companies_csv: Path to companies CSV file
            naics_descriptions_xlsx: Path to NAICS Descriptions file
            naics_index_xlsx: Path to NAICS Index File (keywords/synonyms)
            naics_cross_ref_xlsx: Path to NAICS Cross References file
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
        
        # Debug: Show what codes exist in BEA data
        # print(f"\n📊 BEA Produces data has {len(self.produces_data.index)} industry codes")
        # print(f"Sample codes: {list(self.produces_data.index[:10])}")
        # electronics_codes = [c for c in self.produces_data.index if str(c).startswith('334')]
        # print(f"Electronics manufacturing codes (334xxx): {electronics_codes[:5] if electronics_codes else 'NONE FOUND'}")
        
        # Prepare domestic direct requirements data
        df_requires_domestic = df_requires_domestic.set_index('Code').drop(columns=['Commodity Description'])
        self.requires_domestic_data = df_requires_domestic.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Debug: Show requirement codes
        # print(f"BEA Requirements data has {len(self.requires_domestic_data.columns)} industry codes")
        # print(f"Sample codes: {list(self.requires_domestic_data.columns[:10])}")
        
        # Prepare total requirements data
        df_requires_total = df_requires_total.set_index('Code').drop(columns=['Commodity Description'])
        self.requires_total_data = df_requires_total.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Load NAICS reference files
        self._load_naics_files(naics_descriptions_xlsx, naics_index_xlsx, naics_cross_ref_xlsx)
        
        # Generate produces and requires attributes
        self.produces = self._generate_produces()
        self.requires = self._generate_requires()
    
    def _load_naics_files(self, descriptions_xlsx, index_xlsx, cross_ref_xlsx):
        """Load and prepare NAICS reference files"""
        
        # Load NAICS Descriptions (official ~2k industries)
        try:
            self.naics_descriptions = pd.read_excel(descriptions_xlsx)
            # print(f"Loaded {len(self.naics_descriptions)} NAICS descriptions")
        except Exception as e:
            print(f"Warning: Could not load NAICS descriptions: {e}")
            self.naics_descriptions = pd.DataFrame()
        
        # Load NAICS Index (keywords/synonyms ~10k+ entries)
        try:
            self.naics_index = pd.read_excel(index_xlsx)
            # print(f"Loaded {len(self.naics_index)} NAICS index entries")
        except Exception as e:
            print(f"Warning: Could not load NAICS index: {e}")
            self.naics_index = pd.DataFrame()
        
        # Load Cross References (related industries)
        try:
            self.naics_cross_ref = pd.read_excel(cross_ref_xlsx)
            # print(f"Loaded {len(self.naics_cross_ref)} NAICS cross-references")
        except Exception as e:
            print(f"Warning: Could not load NAICS cross-references: {e}")
            self.naics_cross_ref = pd.DataFrame()
        
        # Build expanded training corpus
        self._build_naics_corpus()
    
    def _build_naics_corpus(self):
        """Combine descriptions and index into expanded training corpus"""
        
        corpus_rows = []
        
        # Add official descriptions
        if not self.naics_descriptions.empty:
            for _, row in self.naics_descriptions.iterrows():
                code = str(row['Code'])
                title = str(row.get('Title', ''))
                description = str(row.get('Description', ''))
                
                # Combine title and description
                full_text = f"{title} {description}".strip()
                
                if full_text and code:
                    corpus_rows.append({
                        'naics_code': code,
                        'text': full_text,
                        'source': 'description'
                    })
        
        # Add index keywords/synonyms (CRITICAL for accuracy)
        if not self.naics_index.empty:
            for _, row in self.naics_index.iterrows():
                # Index file structure: NAICS22 column and INDEX ITEM DESCRIPTION
                code = str(row.get('NAICS22', row.get('Code', '')))
                index_text = str(row.get('INDEX ITEM DESCRIPTION', ''))
                
                if index_text and code and code != 'nan':
                    corpus_rows.append({
                        'naics_code': code,
                        'text': index_text,
                        'source': 'index'
                    })
        
        self.naics_corpus = pd.DataFrame(corpus_rows)
        print(f"Built NAICS corpus with {len(self.naics_corpus)} total entries")
        
        if not self.naics_corpus.empty:
            print(f"  - {len(self.naics_corpus[self.naics_corpus['source']=='description'])} from descriptions")
            print(f"  - {len(self.naics_corpus[self.naics_corpus['source']=='index'])} from index")
    
    def get_company_naics(self, min_confidence=0.15, top_n=10):
        """
        Match company description to NAICS codes using expanded corpus.
        
        Args:
            min_confidence: Minimum cosine similarity threshold
            top_n: Number of top matches to return
        
        Returns:
            list[dict]: Top NAICS matches with code, label, and confidence
        """
        
        # Load company data
        companies = pd.read_csv(self.companies_csv)
        
        if self.ticker not in companies["Ticker"].values:
            raise ValueError(f"Ticker '{self.ticker}' not found in companies.csv.")
        
        company = companies[companies["Ticker"] == self.ticker].iloc[0]
        description = company["Description"]
        
        # Debug: Print the description being matched
        # print(f"\n🔍 Matching description for {self.ticker}:")
        # print(f"{description[:500]}...\n" if len(description) > 500 else f"{description}\n")
        
        if self.naics_corpus.empty:
            print("Warning: NAICS corpus is empty, falling back to Census API")
            return self._fallback_census_api(description, min_confidence)
        
        # TF-IDF matching using expanded corpus
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            max_features=5000
        )
        
        # Fit on corpus
        tfidf_matrix = vectorizer.fit_transform(self.naics_corpus['text'])
        
        # Transform query
        query_vec = vectorizer.transform([description])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Add similarities to corpus
        self.naics_corpus['similarity'] = similarities
        
        # Group by NAICS code and take max similarity (since we have multiple rows per code)
        grouped = self.naics_corpus.groupby('naics_code').agg({
            'similarity': 'max',
            'text': 'first'  # Keep one text example
        }).reset_index()
        
        # FILTER OUT generic administrative/management codes
        # These are 2-digit codes 55 and 56, or codes starting with 551, 561, 562
        generic_prefixes = ['55', '56']
        grouped['is_generic'] = grouped['naics_code'].astype(str).str[:2].isin(generic_prefixes)
        
        # Penalize generic codes heavily
        grouped.loc[grouped['is_generic'], 'similarity'] *= 0.3
        
        # BOOST codes that actually exist in BEA data
        bea_codes = set(str(c) for c in self.produces_data.index)
        grouped['in_bea'] = grouped['naics_code'].astype(str).isin(bea_codes)
        
        # Also check if truncated versions exist (e.g., if BEA has '334' and NAICS match is '334111')
        def has_bea_match(code):
            code_str = str(code)
            # Check exact match
            if code_str in bea_codes:
                return True
            # Check truncated versions
            for length in [5, 4, 3, 2]:
                if len(code_str) >= length and code_str[:length] in bea_codes:
                    return True
            return False
        
        grouped['has_bea_match'] = grouped['naics_code'].apply(has_bea_match)
        
        # MAJOR boost for codes with BEA data (2x)
        grouped.loc[grouped['has_bea_match'], 'similarity'] *= 2.0
        
        # BOOST manufacturing, utilities, and tech codes (3-digit codes 21-33, 51-52)
        priority_prefixes = ['21', '22', '23', '31', '32', '33', '51', '52', '54']
        grouped['is_priority'] = grouped['naics_code'].astype(str).str[:2].isin(priority_prefixes)
        grouped.loc[grouped['is_priority'], 'similarity'] *= 1.2
        
        # Filter and sort
        grouped = grouped[grouped['similarity'] > min_confidence]
        grouped = grouped.sort_values('similarity', ascending=False).head(top_n * 2)  # Get more to filter
        
        if grouped.empty:
            # Fallback to top N if nothing passes threshold
            grouped = self.naics_corpus.groupby('naics_code').agg({
                'similarity': 'max',
                'text': 'first'
            }).reset_index().sort_values('similarity', ascending=False).head(top_n)
        
        # Format output
        results = []
        for _, row in grouped.iterrows():
            # Get full description from descriptions file if available
            full_desc = "N/A"
            if not self.naics_descriptions.empty:
                desc_match = self.naics_descriptions[self.naics_descriptions['Code'].astype(str) == row['naics_code']]
                if not desc_match.empty:
                    full_desc = desc_match.iloc[0].get('Title', row['text'][:100])
            
            if full_desc == "N/A":
                full_desc = row['text'][:100] + '...' if len(row['text']) > 100 else row['text']
            
            results.append({
                'code': row['naics_code'],
                'label': full_desc,
                'confidence': round(float(row['similarity']), 3)
            })
        
        # Filter out generic codes from final results if we have enough priority codes
        priority_results = [r for r in results if not r['code'].startswith(('55', '56'))]
        if len(priority_results) >= 3:
            results = priority_results[:top_n]
        else:
            results = results[:top_n]
        
        # Apply cross-reference boosting if available
        if not self.naics_cross_ref.empty:
            results = self._apply_cross_reference_boost(results)
        
        return results
    
    def _apply_cross_reference_boost(self, results):
        """Boost related industries based on cross-references"""
        
        if not results:
            return results
        
        # Get top match code
        top_code = results[0]['code']
        
        # Find related codes in cross-reference
        related_codes = set()
        for _, row in self.naics_cross_ref.iterrows():
            if str(row.get('Code', '')) == top_code:
                cross_ref_text = str(row.get('Cross-Reference', ''))
                # Extract NAICS codes from cross-reference text (simple regex would work better)
                import re
                found_codes = re.findall(r'\b\d{6}\b', cross_ref_text)
                related_codes.update(found_codes)
        
        # Boost related codes by 10%
        for result in results:
            if result['code'] in related_codes:
                result['confidence'] = min(1.0, result['confidence'] * 1.1)
        
        # Re-sort after boosting
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def _fallback_census_api(self, description, min_confidence):
        """Fallback to original Census API method if NAICS files not available"""
        
        url = "https://api.census.gov/data/2022/cbp?get=NAICS2017,NAICS2017_LABEL,ESTAB,EMP&for=state:*"
        data = requests.get(url).json()
        
        columns = data[0]
        rows = data[1:]
        naics_df = pd.DataFrame(rows, columns=columns)
        naics_df = naics_df.drop_duplicates(subset=["NAICS2017_LABEL"]).dropna(subset=["NAICS2017_LABEL"])
        
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(naics_df["NAICS2017_LABEL"].astype(str))
        
        query_vec = vectorizer.transform([description])
        similarities = cosine_similarity(query_vec, tfidf).flatten()
        
        naics_df["confidence"] = similarities
        filtered = naics_df[naics_df["confidence"] > min_confidence]
        
        if filtered.empty:
            filtered = naics_df.sort_values(by="confidence", ascending=False).head(5)
        else:
            filtered = filtered.sort_values(by="confidence", ascending=False).head(10)
        
        return [
            {
                "code": row["NAICS2017"],
                "label": row["NAICS2017_LABEL"],
                "confidence": round(float(row["confidence"]), 3),
            }
            for _, row in filtered.iterrows()
        ]
    
    def get_top_commodities(self, naics_code, top_n=5):
        """
        Get the top N commodities produced by a given NAICS code.
        Try progressively shorter codes if exact match fails.
        """
        
        # Try exact match first
        code = str(naics_code)
        
        # Try progressively shorter codes (6-digit -> 5-digit -> 4-digit -> 3-digit)
        for length in [6, 5, 4, 3, 2]:
            truncated_code = code[:length] if len(code) >= length else code
            
            # Try as string
            if truncated_code in self.produces_data.index:
                top = self.produces_data.loc[truncated_code].sort_values(ascending=False).head(top_n)
                top = top[top > 0]
                
                results = []
                for commodity_code, production_value in top.items():
                    results.append({
                        'naics_code': str(commodity_code),
                        'production_value': float(production_value),
                        'description': self.exact_descriptions.get(commodity_code, 
                                      self.descriptions.get(commodity_code, 'N/A'))
                    })
                
                if results:  # Only return if we found something
                    return results
            
            # Try as int
            try:
                int_code = int(truncated_code)
                if int_code in self.produces_data.index:
                    top = self.produces_data.loc[int_code].sort_values(ascending=False).head(top_n)
                    top = top[top > 0]
                    
                    results = []
                    for commodity_code, production_value in top.items():
                        results.append({
                            'naics_code': str(commodity_code),
                            'production_value': float(production_value),
                            'description': self.exact_descriptions.get(commodity_code, 
                                          self.descriptions.get(commodity_code, 'N/A'))
                        })
                    
                    if results:
                        return results
            except ValueError:
                continue
        
        return []
    
    def get_top_requirements(self, naics_code, top_n=5, include_indirect=True):
        """
        Get the top N requirements for a given NAICS code.
        Try progressively shorter codes if exact match fails.
        """
        
        code = str(naics_code)
        results = []
        
        # Try progressively shorter codes
        for length in [6, 5, 4, 3, 2]:
            truncated_code = code[:length] if len(code) >= length else code
            
            # Try string version
            if truncated_code in self.requires_domestic_data.columns:
                direct = self.requires_domestic_data[truncated_code].sort_values(ascending=False).head(top_n)
                direct = direct[direct > 0]
                
                for requirement_code, requirement_value in direct.items():
                    results.append({
                        'naics_code': str(requirement_code),
                        'requirement_value': float(requirement_value),
                        'layer': 'direct',
                        'description': self.exact_descriptions.get(requirement_code,
                                      self.requirements_descriptions.get(requirement_code, 'N/A'))
                    })
                
                # Indirect requirements
                if include_indirect and truncated_code in self.requires_total_data.columns:
                    total_series = self.requires_total_data[truncated_code]
                    direct_series = self.requires_domestic_data[truncated_code]
                    
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
                
                if results:  # Found something
                    return results
            
            # Try int version
            try:
                int_code = int(truncated_code)
                if int_code in self.requires_domestic_data.columns:
                    direct = self.requires_domestic_data[int_code].sort_values(ascending=False).head(top_n)
                    direct = direct[direct > 0]
                    
                    for requirement_code, requirement_value in direct.items():
                        results.append({
                            'naics_code': str(requirement_code),
                            'requirement_value': float(requirement_value),
                            'layer': 'direct',
                            'description': self.exact_descriptions.get(requirement_code,
                                          self.requirements_descriptions.get(requirement_code, 'N/A'))
                        })
                    
                    if include_indirect and int_code in self.requires_total_data.columns:
                        total_series = self.requires_total_data[int_code]
                        direct_series = self.requires_domestic_data[int_code]
                        
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
                    
                    if results:
                        return results
            except ValueError:
                continue
        
        return results
    
    def _merge_duplicates(self, items_list):
        """Merge duplicate NAICS codes by summing values and averaging confidence"""
        
        df = pd.DataFrame(items_list)
        if df.empty:
            return []
        
        value_col = 'production_value' if 'production_value' in df.columns else 'requirement_value'
        
        grouped = df.groupby('naics_code').agg({
            value_col: 'sum',
            'confidence': 'mean',
            'description': 'first',
            **({'layer': 'first'} if 'layer' in df.columns else {})
        }).reset_index()
        
        return grouped.to_dict('records')
    
    def _normalize_values(self, items_list, by_layer=False):
        """Normalize production/requirement values to sum to 1"""
        
        if not items_list:
            return []

        value_col = 'production_value' if 'production_value' in items_list[0] else 'requirement_value'

        if by_layer:
            from collections import defaultdict
            layer_groups = defaultdict(list)
            for item in items_list:
                layer = item.get('layer', 'combined')
                layer_groups[layer].append(item)
            
            for layer, group in layer_groups.items():
                total = sum(i[value_col] for i in group)
                if total > 0:
                    for i in group:
                        i[value_col] /= total
            
            normalized_items = [i for group in layer_groups.values() for i in group]
            return normalized_items
        else:
            total = sum(item[value_col] for item in items_list)
            if total > 0:
                for item in items_list:
                    item[value_col] /= total
            return items_list

    def _generate_produces(self):
        """Generate produces attribute with improved NAICS matching"""
        
        company_naics = self.get_company_naics()
        all_commodities = []
        
        for naics_match in company_naics:
            parent_code = naics_match['code']
            parent_confidence = naics_match['confidence']
            
            commodities = self.get_top_commodities(parent_code, top_n=5)
            
            for commodity in commodities:
                all_commodities.append({
                    'naics_code': commodity['naics_code'],
                    'production_value': commodity['production_value'],
                    'confidence': parent_confidence,
                    'description': commodity['description']
                })
        
        all_commodities = self._merge_duplicates(all_commodities)
        all_commodities = self._normalize_values(all_commodities)
        
        return all_commodities
    
    def _generate_requires(self):
        """Generate requires attribute with improved NAICS matching"""
        
        company_naics = self.get_company_naics()
        # print("NAICS matches:", company_naics)
        all_requirements = []
        
        for naics_match in company_naics:
            parent_code = naics_match['code']
            parent_confidence = naics_match['confidence']
            
            requirements = self.get_top_requirements(parent_code, top_n=5, include_indirect=True)
            
            for requirement in requirements:
                all_requirements.append({
                    'naics_code': requirement['naics_code'],
                    'requirement_value': requirement['requirement_value'],
                    'confidence': parent_confidence,
                    'layer': requirement['layer'],
                    'description': requirement['description']
                })
        
        all_requirements = self._merge_duplicates(all_requirements)
        all_requirements = self._normalize_values(all_requirements, by_layer=True)
        
        return all_requirements


# Example usage
if __name__ == "__main__":
    commodities = Commodities("TSLA")
    
    print("\nPRODUCES (merged & normalized):")
    for item in commodities.produces:
        print(f"  {item['naics_code']}: {item['production_value']:.6f} (conf: {item['confidence']:.3f}) - {item['description']}")
    
    print(f"\nTotal produces sum: {sum(i['production_value'] for i in commodities.produces):.6f}")
    
    print("\nREQUIRES (merged & normalized, direct + indirect):")
    for item in commodities.requires:
        layer = item.get('layer', 'combined')
        print(f"  {item['naics_code']}: {item['requirement_value']:.6f} (conf: {item['confidence']:.3f}) [{layer}] - {item['description']}")
    
    print(f"\nTotal requires sum: {sum(i['requirement_value'] for i in commodities.requires):.6f}")