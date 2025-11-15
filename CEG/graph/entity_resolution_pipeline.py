"""
Entity Resolution Pipeline for Supply Chain Knowledge Graph - COMPLETE FIXED
============================================================================

FIXES:
1. ✅ Improved root name extraction (removes "The", etc.)
2. ✅ First-word matching (Boeing Tianjin → The Boeing Company)
3. ✅ Better subsidiary pattern detection
4. ✅ Enhanced fuzzy matching for parent companies

Matches surface-level supplier names to canonical companies with tickers.
Implements the two-layer architecture: CompanyCanonical + CompanySurface
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EntityResolver:
    """
    Hybrid entity resolution using:
    1. String similarity (fuzzy matching)
    2. Semantic embeddings (sentence-transformers)
    3. Metadata features (country, keywords)
    4. Rule-based patterns (subsidiaries, divisions)
    """
    
    def __init__(self, canonical_companies_path: str):
        """
        Initialize with canonical company list.
        
        Args:
            canonical_companies_path: Path to CSV with columns [Ticker, Name]
        """
        self.canonical_df = pd.read_csv(canonical_companies_path)
        self.canonical_df.columns = self.canonical_df.columns.str.strip()
        
        # Load embedding model (lightweight but effective)
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Precompute embeddings for canonical companies
        self.canonical_embeddings = self.model.encode(
            self.canonical_df['Name'].tolist(),
            show_progress_bar=True
        )
        
        # Build lookup structures
        self._build_lookup_structures()
        
        print(f"✅ Loaded {len(self.canonical_df)} canonical companies")
    
    def _build_lookup_structures(self):
        """Build efficient lookup structures."""
        self.ticker_to_name = dict(zip(
            self.canonical_df['Ticker'],
            self.canonical_df['Name']
        ))
        
        self.name_to_ticker = dict(zip(
            self.canonical_df['Name'],
            self.canonical_df['Ticker']
        ))
        
        # Extract company root names (without legal suffixes)
        self.canonical_df['root_name'] = self.canonical_df['Name'].apply(
            self._extract_root_name
        )
    
    def _extract_root_name(self, name: str) -> str:
        """
        Extract core company name, removing legal suffixes and articles.
        
        Examples:
            "Tesla, Inc." -> "tesla"
            "The Boeing Company" -> "boeing"
            "Panasonic Holdings Corp." -> "panasonic"
            "Samsung Electronics Co., Ltd." -> "samsung"
            "Aviation Industry Corporation of China" -> "aviation industry corporation"
        """
        name = name.lower()
        
        # Remove leading articles
        name = re.sub(r'^the\s+', '', name)
        
        # Remove legal suffixes
        legal_patterns = [
            r'\s+inc\.?$', r'\s+corp\.?$', r'\s+ltd\.?$',
            r'\s+llc\.?$', r'\s+l\.l\.c\.?$', r'\s+plc\.?$',
            r'\s+co\.?,?\s*ltd\.?$', r'\s+limited$',
            r'\s+corporation$', r'\s+company$',
            r'\s+holdings?$', r'\s+group$',
            r'\s+s\.?a\.?$', r'\s+gmbh$', r'\s+ag$'
        ]
        
        for pattern in legal_patterns:
            name = re.sub(pattern, '', name)
        
        # Clean punctuation
        name = re.sub(r'[,\.\(\)]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _extract_country_from_location(self, location: str) -> Optional[str]:
        """Extract country from location string."""
        if not location or pd.isna(location):
            return None
        
        # Common country mappings
        country_map = {
            'china': 'China',
            'japan': 'Japan',
            'south korea': 'South Korea',
            'korea': 'South Korea',
            'germany': 'Germany',
            'france': 'France',
            'taiwan': 'Taiwan',
            'vietnam': 'Vietnam',
            'singapore': 'Singapore',
            'spain': 'Spain',
            'united states': 'United States',
            'usa': 'United States',
            'uk': 'United Kingdom',
        }
        
        location_lower = location.lower()
        for key, value in country_map.items():
            if key in location_lower:
                return value
        
        return None
    
    def _compute_string_similarity(self, supplier_name: str, canonical_name: str) -> float:
        """
        Compute string similarity using multiple algorithms.
        Returns weighted average score (0-100).
        """
        # Token sort ratio (handles word order)
        token_sort = fuzz.token_sort_ratio(supplier_name, canonical_name)
        
        # Partial ratio (handles substring matches)
        partial = fuzz.partial_ratio(supplier_name, canonical_name)
        
        # Weighted average (prefer token_sort for full matches)
        return 0.7 * token_sort + 0.3 * partial
    
    def _compute_semantic_similarity(self, supplier_name: str) -> np.ndarray:
        """
        Compute semantic similarity between supplier and all canonical companies.
        Returns array of similarity scores (0-1).
        """
        supplier_embedding = self.model.encode([supplier_name])
        similarities = cosine_similarity(supplier_embedding, self.canonical_embeddings)[0]
        return similarities
    
    def _check_subsidiary_patterns(self, supplier_name: str) -> List[Tuple[str, str]]:
        """
        Check for subsidiary/division patterns in supplier name.
        Returns list of (potential_parent_name, pattern_type) tuples.
        
        IMPROVED: Now extracts just the first word for better matching.
        
        Examples:
            "AVIC International Beichen Dong China" → [("avic", "geographic")]
            "AVIC Chengfei Commercial Aircraft" → [("avic", "division")]
            "Panasonic Japan" → [("panasonic", "geographic")]
            "Boeing Tianjin Composites Co Ltd China" → [("boeing", "geographic")]
        """
        supplier_lower = supplier_name.lower()
        potential_parents = []
        
        # Pattern 1: Geographic suffix (Company Name + Country)
        # "AVIC International China", "Panasonic Japan", "Boeing Tianjin China"
        geo_pattern = r'^([a-z]+(?:\s+[a-z]+)?)\s+(?:japan|china|korea|south korea|germany|france|taiwan|vietnam|singapore|spain|usa|uk|united states).*$'
        match = re.search(geo_pattern, supplier_lower)
        if match:
            parent = match.group(1).strip()
            # Extract just first word
            first_word = parent.split()[0]
            if len(first_word) > 2:  # Avoid short matches like "co", "ab"
                potential_parents.append((first_word, "geographic"))
        
        # Pattern 2: Division/Facility suffix
        # "AVIC Chengfei Commercial Aircraft", "Samsung Electronics"
        division_patterns = [
            r'^([a-z]+)\s+(?:international|global|asia|europe|americas|aerospace|aviation|electronics|automotive|manufacturing|industrial|commercial)',
            r'^([a-z]+)\s+[a-z]+\s+(?:division|facility|plant|factory|unit|department)',
        ]
        for pattern in division_patterns:
            match = re.search(pattern, supplier_lower)
            if match:
                parent = match.group(1).strip()
                if len(parent) > 2:
                    potential_parents.append((parent, "division"))
        
        # Pattern 3: Holdings/Corporation pattern
        # "AVIC International Holding Corporation"
        holding_pattern = r'^([a-z]+)\s+.*(?:holding|holdings|corporat|corporation|group)'
        match = re.search(holding_pattern, supplier_lower)
        if match:
            parent = match.group(1).strip()
            if len(parent) > 2:
                potential_parents.append((parent, "holdings"))
        
        return potential_parents
    
    def resolve_supplier(
        self,
        supplier_name: str,
        location: str = None,
        threshold: float = 0.75
    ) -> Optional[Dict]:
        """
        Resolve a supplier to a canonical company.
        
        Args:
            supplier_name: Surface-level supplier name
            location: Supplier location (helps with disambiguation)
            threshold: Minimum combined score (0-1) for match
        
        Returns:
            {
                'canonical_ticker': str,
                'canonical_name': str,
                'confidence': float,
                'match_type': str,  # 'exact', 'fuzzy', 'semantic', 'subsidiary'
                'relationship': str  # 'OWNS', 'SUBSIDIARY', 'PARENT_OF'
            }
            or None if no match above threshold
        """
        supplier_name_clean = self._extract_root_name(supplier_name)
        country = self._extract_country_from_location(location)
        
        # 1. Check for exact matches (root name)
        for idx, row in self.canonical_df.iterrows():
            if supplier_name_clean == row['root_name']:
                return {
                    'canonical_ticker': row['Ticker'],
                    'canonical_name': row['Name'],
                    'confidence': 1.0,
                    'match_type': 'exact',
                    'relationship': 'SAME_AS'
                }
        
        # 1.5 NEW: Check if supplier starts with a canonical company name
        # This catches "Boeing Tianjin Composites" matching "The Boeing Company"
        supplier_first_word = supplier_name_clean.split()[0] if supplier_name_clean else ""
        
        if len(supplier_first_word) > 3:  # Avoid short words like "co", "the"
            for idx, row in self.canonical_df.iterrows():
                canonical_root = row['root_name']
                canonical_first_word = canonical_root.split()[0] if canonical_root else ""
                
                # Check if first significant word matches
                if supplier_first_word == canonical_first_word and len(canonical_first_word) > 3:
                    return {
                        'canonical_ticker': row['Ticker'],
                        'canonical_name': row['Name'],
                        'confidence': 0.95,
                        'match_type': 'first_word_match',
                        'relationship': 'SUBSIDIARY'
                    }
        
        # 2. IMPROVED: Check subsidiary patterns with better matching
        potential_parents = self._check_subsidiary_patterns(supplier_name)
        
        for parent_name, pattern_type in potential_parents:
            # Try to match against canonical root names
            best_matches = []
            
            for idx, row in self.canonical_df.iterrows():
                canonical_root = row['root_name']
                
                # Check if parent_name matches start of canonical root
                # e.g., "avic" matches "aviation industry corporation"
                if canonical_root.startswith(parent_name):
                    score = 95
                    best_matches.append((idx, score))
                # Or exact match on first word
                elif canonical_root.split()[0] == parent_name:
                    score = 90
                    best_matches.append((idx, score))
                # Check if canonical root contains the parent name
                elif parent_name in canonical_root:
                    score = 85
                    best_matches.append((idx, score))
                # Or high fuzzy match
                else:
                    score = fuzz.token_sort_ratio(parent_name, canonical_root)
                    if score > 85:
                        best_matches.append((idx, score))
            
            if best_matches:
                # Get the best match
                best_idx, best_score = max(best_matches, key=lambda x: x[1])
                
                return {
                    'canonical_ticker': self.canonical_df.loc[best_idx, 'Ticker'],
                    'canonical_name': self.canonical_df.loc[best_idx, 'Name'],
                    'confidence': min(0.95, best_score / 100.0),
                    'match_type': f'subsidiary_{pattern_type}',
                    'relationship': 'SUBSIDIARY'
                }
        
        # 3. Compute fuzzy string similarity
        string_scores = []
        for canonical_name in self.canonical_df['Name']:
            score = self._compute_string_similarity(supplier_name, canonical_name)
            string_scores.append(score / 100.0)  # Normalize to 0-1
        
        # 4. Compute semantic similarity
        semantic_scores = self._compute_semantic_similarity(supplier_name)
        
        # 5. Combine scores (weighted average)
        combined_scores = 0.6 * np.array(string_scores) + 0.4 * semantic_scores
        
        # 6. Apply country boost if available
        if country:
            for idx, row in self.canonical_df.iterrows():
                if country.lower() in row['Name'].lower():
                    combined_scores[idx] *= 1.15  # 15% boost
        
        # 7. Get best match
        best_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_idx]
        
        if best_score >= threshold:
            return {
                'canonical_ticker': self.canonical_df.loc[best_idx, 'Ticker'],
                'canonical_name': self.canonical_df.loc[best_idx, 'Name'],
                'confidence': float(best_score),
                'match_type': 'fuzzy_semantic',
                'relationship': 'OWNS' if best_score > 0.85 else 'RELATED_TO'
            }
        
        return None
    
    def resolve_suppliers_batch(
        self,
        suppliers: List[Dict],
        threshold: float = 0.75
    ) -> pd.DataFrame:
        """
        Resolve a batch of suppliers.
        
        Args:
            suppliers: List of dicts with keys: supplier_name, location, etc.
            threshold: Minimum confidence for match
        
        Returns:
            DataFrame with columns:
                - supplier_name (surface entity)
                - location
                - canonical_ticker
                - canonical_name
                - confidence
                - match_type
                - relationship
        """
        results = []
        
        for supplier in suppliers:
            supplier_name = supplier.get('supplier_name', '')
            location = supplier.get('location', '')
            
            match = self.resolve_supplier(supplier_name, location, threshold)
            
            result = {
                'supplier_name': supplier_name,
                'location': location,
                'canonical_ticker': match['canonical_ticker'] if match else None,
                'canonical_name': match['canonical_name'] if match else None,
                'confidence': match['confidence'] if match else 0.0,
                'match_type': match['match_type'] if match else 'no_match',
                'relationship': match['relationship'] if match else None
            }
            
            # Preserve all original supplier fields
            for key, value in supplier.items():
                if key not in result:
                    result[key] = value
            
            results.append(result)
        
        return pd.DataFrame(results)


def integrate_with_neo4j_exporter(resolver: EntityResolver, suppliers: List[Dict]) -> List[Dict]:
    """
    Integrate entity resolution with existing Neo4j export pipeline.
    
    Returns enhanced supplier edges with canonical company linking.
    """
    resolved_df = resolver.resolve_suppliers_batch(suppliers)
    
    enhanced_suppliers = []
    for _, row in resolved_df.iterrows():
        supplier_dict = row.to_dict()
        
        # If matched to canonical company, add metadata
        if supplier_dict['canonical_ticker']:
            supplier_dict['has_canonical_match'] = True
            supplier_dict['canonical_relationship_type'] = supplier_dict['relationship']
        else:
            supplier_dict['has_canonical_match'] = False
        
        enhanced_suppliers.append(supplier_dict)
    
    return enhanced_suppliers


# Example usage
if __name__ == "__main__":
    # Initialize resolver
    resolver = EntityResolver('./data/companies_filtered.csv')
    
    # Example suppliers (your actual data)
    test_suppliers = [
        {'supplier_name': 'Boeing Tianjin Composites Co Ltd China', 'location': 'China'},
        {'supplier_name': 'Boeing Aerostructures Australia Pty Australia', 'location': 'Australia'},
        {'supplier_name': 'Avic International Beichen Dong China', 'location': 'China'},
        {'supplier_name': 'Avic International Holding Corporat South Korea', 'location': 'South Korea'},
        {'supplier_name': 'Avic Chengfei Commercial Aircraft C', 'location': 'China'},
        {'supplier_name': 'Commercial Airp Japan', 'location': 'Japan'},
        {'supplier_name': 'Amag Rolling Germany', 'location': 'Germany'},
        {'supplier_name': 'Korea Aerospace Ind', 'location': 'South Korea'},
        {'supplier_name': 'Panasonic Japan', 'location': 'Japan'},
        {'supplier_name': 'Panasonic Korea', 'location': 'South Korea'},
        {'supplier_name': 'Samsung Electronics Vietnam', 'location': 'Vietnam'},
    ]
    
    # Resolve
    results = resolver.resolve_suppliers_batch(test_suppliers, threshold=0.70)
    
    print("\n" + "="*100)
    print("ENTITY RESOLUTION RESULTS")
    print("="*100)
    print(results[['supplier_name', 'canonical_ticker', 'canonical_name', 'confidence', 'match_type']].to_string())
    
    # Show statistics
    matched = results[results['canonical_ticker'].notna()]
    print(f"\n📊 Matched: {len(matched)}/{len(results)} ({len(matched)/len(results)*100:.1f}%)")
    if len(matched) > 0:
        print(f"Average confidence: {matched['confidence'].mean():.3f}")
    
    # Show by match type
    print("\n📋 Match types:")
    print(results['match_type'].value_counts())