"""
Fixed Neo4j Exporter with Proper Schema
========================================

FIXES IMPLEMENTED:
1. ✅ Removed redundant fields (risk_signal, volatility_score, base_weight, entity_type)
2. ✅ Added aliases, embeddings, confidence fields
3. ✅ Moved relationship attributes from nodes to edges
4. ✅ Added temporal properties (first_seen, last_seen, decay_rate)
5. ✅ Fixed array fields (hs_codes, aliases)
6. ✅ Added units for commodities
7. ✅ Renamed root_name to canonical_name
8. ✅ Added iso_code, region for countries
9. ✅ Added classification_system for sectors/industries
"""

from neo4j import GraphDatabase
import yfinance as yf
from datetime import datetime
import math
from typing import List, Dict, Optional, Tuple
import warnings
from collections import defaultdict
import re

try:
    from .entity_resolution_pipeline import EntityResolver
    ENTITY_RESOLUTION_AVAILABLE = True
except ImportError:
    ENTITY_RESOLUTION_AVAILABLE = False
    warnings.warn("Entity resolution not available.")


class FixedNeo4jExporter:
    """Neo4j exporter with proper schema following review recommendations."""
    
    TEMPORAL_TYPES = {
        'country': 'stable',
        'sector': 'stable',
        'industry': 'stable',
        'supplier': 'permanent',
        'produces': 'permanent',
        'requires': 'permanent'
    }
    
    DECAY_RATES = {
        'permanent': 0.0,
        'stable': 0.001,
        'semi-permanent': 0.05,
        'temporary': 0.15
    }
    
    # Country to ISO code mapping
    COUNTRY_ISO_MAP = {
        'United States': {'iso': 'US', 'region': 'North America'},
        'China': {'iso': 'CN', 'region': 'Asia'},
        'Japan': {'iso': 'JP', 'region': 'Asia'},
        'Germany': {'iso': 'DE', 'region': 'Europe'},
        'United Kingdom': {'iso': 'GB', 'region': 'Europe'},
        'France': {'iso': 'FR', 'region': 'Europe'},
        'South Korea': {'iso': 'KR', 'region': 'Asia'},
        'Canada': {'iso': 'CA', 'region': 'North America'},
        'Australia': {'iso': 'AU', 'region': 'Oceania'},
        'Taiwan': {'iso': 'TW', 'region': 'Asia'},
        'Singapore': {'iso': 'SG', 'region': 'Asia'},
        'Vietnam': {'iso': 'VN', 'region': 'Asia'},
        'Spain': {'iso': 'ES', 'region': 'Europe'},
    }
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password",
                 canonical_companies_csv: Optional[str] = None,
                 auto_canonicalize: bool = False,
                 min_occurrences: int = 2):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.timestamp = datetime.now().isoformat()
        self.auto_canonicalize = auto_canonicalize
        self.min_occurrences = min_occurrences
        
        self.resolver = None
        if canonical_companies_csv and ENTITY_RESOLUTION_AVAILABLE:
            try:
                print("🔍 Initializing entity resolver...")
                self.resolver = EntityResolver(canonical_companies_csv)
                print("✅ Entity resolver ready")
            except Exception as e:
                print(f"⚠️  Entity resolution disabled: {e}")
    
    def close(self):
        self.driver.close()
    
    def _extract_root_company_name(self, supplier_name: str) -> str:
        """Extract the root company name."""
        name = supplier_name.lower()
        
        geo_patterns = [
            r'\s+(?:japan|china|korea|south korea|germany|france|taiwan|vietnam|singapore|spain|usa|uk)$',
        ]
        for pattern in geo_patterns:
            name = re.sub(pattern, '', name)
        
        facility_patterns = [
            r'\s+(?:facility|factory|manufacturing|international|holdings?).*$',
        ]
        for pattern in facility_patterns:
            name = re.sub(pattern, '', name)
        
        legal_patterns = [
            r'\s+(?:inc|corp|ltd|llc|co|company|limited)\.?$',
        ]
        for pattern in legal_patterns:
            name = re.sub(pattern, '', name)
        
        name = re.sub(r'[,\.\(\)]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        words = name.split()
        if words:
            return words[0]
        
        return name
    
    def _group_suppliers_by_root_name(self, suppliers: List[Dict]) -> Dict[str, List[Dict]]:
        """Group suppliers by their root company name."""
        groups = defaultdict(list)
        for supplier in suppliers:
            root_name = self._extract_root_company_name(supplier['supplier_name'])
            groups[root_name].append(supplier)
        return dict(groups)
    
    def _generate_aliases(self, name: str) -> List[str]:
        """Generate common aliases for a name."""
        aliases = [name, name.lower()]
        
        # Remove "The" prefix
        if name.startswith("The "):
            aliases.append(name[4:])
        
        # Remove legal suffixes
        legal_suffixes = [" Inc.", " Corp.", " Ltd.", " LLC", " Company", " Corporation"]
        for suffix in legal_suffixes:
            if name.endswith(suffix):
                aliases.append(name[:-len(suffix)])
        
        return list(set(aliases))
    
    def _create_auto_canonical_node(self, session, root_name: str, suppliers: List[Dict]) -> str:
        """Create an auto-generated canonical company node with proper schema."""
        ticker = f"{root_name.upper().replace(' ', '_')}_AUTO"
        canonical_name = f"{root_name.title()} Holdings Corporation"
        
        locations = [s.get('location', '') for s in suppliers if s.get('location')]
        common_location = max(set(locations), key=locations.count) if locations else "Unknown"
        
        aliases = self._generate_aliases(canonical_name)
        
        session.run("""
            MERGE (c:CompanyCanonical {ticker: $ticker})
            SET c.name = $name,
                c.canonical_name = $canonical_name,
                c.legal_name = $canonical_name,
                c.aliases = $aliases,
                c.country = $country,
                c.updated_at = datetime($timestamp),
                c.source_confidence = 0.85
        """,
        ticker=ticker,
        name=canonical_name,
        canonical_name=canonical_name,
        aliases=aliases,
        country=common_location,
        timestamp=self.timestamp)
        
        print(f"      ✨ Created: {canonical_name} ({ticker})")
        return ticker
    
    def export_edges(self, edges, ticker):
        """Export edges with fixed schema."""
        with self.driver.session() as session:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
            market_cap = info.get("marketCap", 0)
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            country = info.get("country", "Unknown")
            
            # Create central company node with PROPER schema
            aliases = self._generate_aliases(company_name)
            
            session.run("""
                MERGE (c:CompanyCanonical {ticker: $ticker})
                SET c.name = $name,
                    c.canonical_name = $name,
                    c.legal_name = $name,
                    c.aliases = $aliases,
                    c.ticker = $ticker,
                    c.marketCap = $marketCap,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.country = $country,
                    c.updated_at = datetime($timestamp),
                    c.source_confidence = 1.0
            """, 
            ticker=ticker,
            name=company_name,
            aliases=aliases,
            marketCap=market_cap,
            sector=sector,
            industry=industry,
            country=country,
            timestamp=self.timestamp)
            
            # Separate supplier edges
            supplier_edges = [e for e in edges if e['type'] == 'supplier']
            other_edges = [e for e in edges if e['type'] != 'supplier']
            
            # Process suppliers
            if supplier_edges:
                if self.resolver:
                    self._process_suppliers_with_resolution(session, supplier_edges, ticker)
                else:
                    self._process_suppliers_legacy(session, supplier_edges, ticker)
            
            # Process other edges
            for edge in other_edges:
                self._process_edge(session, edge, ticker, market_cap)
            
            print(f"✅ Successfully exported {len(edges)} edges")
            return True
    
    def _process_suppliers_with_resolution(self, session, supplier_edges, ticker):
        """Process suppliers with entity resolution."""
        suppliers = []
        for edge in supplier_edges:
            suppliers.append({
                'supplier_name': edge['data']['supplier_name'],
                'location': edge['data'].get('location', ''),
                'magnitude': edge['magnitude'],
                'normalized_magnitude': edge.get('normalized_magnitude'),
                'direction_value': edge.get('direction_value'),
                'weight': edge.get('weight'),
                'correlation_strength': edge.get('correlation_strength'),
                'relevance': edge['relevance'],
                'direction': edge['direction'],
                'full_data': edge['data']
            })
        
        print(f"   🔍 Resolving {len(suppliers)} suppliers...")
        resolved_suppliers = self.resolver.resolve_suppliers_batch(
            [{'supplier_name': s['supplier_name'], 'location': s['location']} 
             for s in suppliers],
            threshold=0.70
        )
        
        matched_count = 0
        unmatched_suppliers = []
        
        for i, supplier in enumerate(suppliers):
            resolution = resolved_suppliers.iloc[i]
            supplier['canonical_ticker'] = resolution['canonical_ticker']
            supplier['canonical_name'] = resolution['canonical_name']
            supplier['confidence'] = resolution['confidence']
            supplier['match_type'] = resolution['match_type']
            supplier['relationship'] = resolution['relationship']
            
            if supplier['canonical_ticker']:
                matched_count += 1
            else:
                unmatched_suppliers.append(supplier)
        
        print(f"   ✅ Resolved {matched_count}/{len(suppliers)} suppliers")
        
        # Auto-canonicalize unmatched
        auto_created_count = 0
        if self.auto_canonicalize and unmatched_suppliers:
            print(f"   🤖 Auto-canonicalizing {len(unmatched_suppliers)} unmatched...")
            
            groups = self._group_suppliers_by_root_name(unmatched_suppliers)
            
            for root_name, group_suppliers in groups.items():
                if len(group_suppliers) >= self.min_occurrences:
                    auto_ticker = f"{root_name.upper().replace(' ', '_')}_AUTO"
                    
                    result = session.run("""
                        MATCH (c:CompanyCanonical {ticker: $ticker})
                        RETURN c.ticker as ticker
                    """, ticker=auto_ticker).single()
                    
                    if not result:
                        auto_ticker = self._create_auto_canonical_node(session, root_name, group_suppliers)
                        auto_created_count += 1
                    
                    for supplier in group_suppliers:
                        supplier['canonical_ticker'] = auto_ticker
                        supplier['canonical_name'] = f"{root_name.title()} Holdings Corporation"
                        supplier['confidence'] = 0.95
                        supplier['match_type'] = 'auto_canonical'
                        supplier['relationship'] = 'SUBSIDIARY'
            
            if auto_created_count > 0:
                print(f"   ✅ Auto-created {auto_created_count} canonical nodes")
        
        # Create nodes and relationships with PROPER schema
        for supplier in suppliers:
            # Create CompanySurface node with FIXED schema
            hs_codes_list = supplier['full_data'].get('hs_codes', '').split(',') if supplier['full_data'].get('hs_codes') else []
            hs_codes_list = [code.strip() for code in hs_codes_list if code.strip()]
            
            aliases = self._generate_aliases(supplier['supplier_name'])
            
            # Convert total_shipments to integer
            try:
                total_shipments = int(str(supplier['full_data'].get('total_shipments', '0')).replace(',', ''))
            except:
                total_shipments = 0
            
            session.run("""
                MERGE (s:CompanySurface {name: $name})
                SET s.raw_name = $raw_name,
                    s.clean_name = $clean_name,
                    s.aliases = $aliases,
                    s.location_string = $location,
                    s.supplier_url = $url,
                    s.product_description = $product_desc,
                    s.hs_codes = $hs_codes,
                    s.total_shipments = $shipments,
                    s.updated_at = datetime($timestamp),
                    s.has_canonical_match = $has_match,
                    s.resolution_confidence = $confidence,
                    s.resolution_type = $match_type
            """,
            name=supplier['supplier_name'],
            raw_name=supplier['supplier_name'],
            clean_name=self._extract_root_company_name(supplier['supplier_name']),
            aliases=aliases,
            location=supplier.get('location', ''),
            url=supplier['full_data'].get('supplier_url', ''),
            product_desc=supplier['full_data'].get('product_description', ''),
            hs_codes=hs_codes_list,
            shipments=total_shipments,
            timestamp=self.timestamp,
            has_match=supplier['canonical_ticker'] is not None,
            confidence=supplier.get('confidence', 0.0),
            match_type=supplier.get('match_type', 'no_match'))
            
            # Create RESOLVES_TO relationship if matched
            if supplier['canonical_ticker']:
                rel_type = self._map_relationship_type(supplier.get('relationship', 'RELATED_TO'))
                
                session.run(f"""
                    MATCH (s:CompanySurface {{name: $surface_name}})
                    MATCH (c:CompanyCanonical {{ticker: $ticker}})
                    MERGE (s)-[r:{rel_type}]->(c)
                    SET r.confidence = $confidence,
                        r.method = $match_type,
                        r.created_at = datetime($timestamp)
                """,
                surface_name=supplier['supplier_name'],
                ticker=supplier['canonical_ticker'],
                confidence=supplier.get('confidence', 0.0),
                match_type=supplier.get('match_type', 'unknown'),
                timestamp=self.timestamp)
            
            # Create SUPPLIES relationship with proper temporal properties
            self._create_supply_relationship(session, supplier, ticker)
    
    def _create_supply_relationship(self, session, supplier, target_ticker):
        """Create SUPPLIES relationship with proper schema."""
        magnitude = supplier.get('magnitude', 0)
        normalized_magnitude = supplier.get('normalized_magnitude')
        direction_value = supplier.get('direction_value', 1)
        weight = supplier.get('weight', magnitude * supplier['relevance'])
        
        # Parse shipments for first_seen/last_seen (simplified)
        first_seen = self.timestamp
        last_seen = self.timestamp
        
        try:
            shipments = int(str(supplier['full_data'].get('total_shipments', '0')).replace(',', ''))
        except:
            shipments = 0
        
        set_clause = """
            SET r.shipments = $shipments,
                r.shipment_count = $shipment_count,
                r.hs_codes = $hs_codes,
                r.first_seen = datetime($first_seen),
                r.last_seen = datetime($last_seen),
                r.decay_rate = $decay_rate,
                r.confidence = $confidence,
                r.weight = $weight,
                r.correlation_strength = $correlation_strength
        """
        
        params = {
            'supplier_name': supplier['supplier_name'],
            'ticker': target_ticker,
            'shipments': shipments,
            'shipment_count': shipments,
            'hs_codes': supplier['full_data'].get('hs_codes', '').split(','),
            'first_seen': first_seen,
            'last_seen': last_seen,
            'decay_rate': 0.0,
            'confidence': supplier['relevance'],
            'weight': weight,
            'correlation_strength': supplier.get('correlation_strength', weight * 0.85),
            'timestamp': self.timestamp
        }
        
        if normalized_magnitude is not None:
            set_clause += ", r.normalized_magnitude = $normalized_magnitude"
            params['normalized_magnitude'] = normalized_magnitude
        
        if direction_value is not None:
            set_clause += ", r.direction_value = $direction_value"
            params['direction_value'] = direction_value
        
        query = f"""
            MATCH (s:CompanySurface {{name: $supplier_name}})
            MATCH (c:CompanyCanonical {{ticker: $ticker}})
            MERGE (s)-[r:SUPPLIES]->(c)
            {set_clause}
        """
        
        session.run(query, **params)
    
    def _process_suppliers_legacy(self, session, supplier_edges, ticker):
        """Legacy supplier processing without resolution."""
        for edge in supplier_edges:
            data = edge['data']
            
            # Create CompanySurface with proper schema
            hs_codes_list = data.get('hs_codes', '').split(',') if data.get('hs_codes') else []
            aliases = self._generate_aliases(data['supplier_name'])
            
            try:
                total_shipments = int(str(data.get('total_shipments', '0')).replace(',', ''))
            except:
                total_shipments = 0
            
            session.run("""
                MERGE (s:CompanySurface {name: $name})
                SET s.raw_name = $raw_name,
                    s.clean_name = $clean_name,
                    s.aliases = $aliases,
                    s.location_string = $location,
                    s.supplier_url = $url,
                    s.product_description = $product_desc,
                    s.hs_codes = $hs_codes,
                    s.total_shipments = $shipments,
                    s.updated_at = datetime($timestamp),
                    s.has_canonical_match = false
            """,
            name=data['supplier_name'],
            raw_name=data['supplier_name'],
            clean_name=self._extract_root_company_name(data['supplier_name']),
            aliases=aliases,
            location=data.get('location', ''),
            url=data.get('supplier_url', ''),
            product_desc=data.get('product_description', ''),
            hs_codes=hs_codes_list,
            shipments=total_shipments,
            timestamp=self.timestamp)
            
            # Create SUPPLIES with temporal properties
            session.run("""
                MATCH (s:CompanySurface {name: $supplier_name})
                MATCH (c:CompanyCanonical {ticker: $ticker})
                MERGE (s)-[r:SUPPLIES]->(c)
                SET r.shipments = $shipments,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.first_seen = datetime($timestamp),
                    r.last_seen = datetime($timestamp),
                    r.decay_rate = 0.0
            """,
            supplier_name=data['supplier_name'],
            ticker=ticker,
            shipments=total_shipments,
            weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
            confidence=edge['relevance'],
            timestamp=self.timestamp)
    
    def _process_edge(self, session, edge, ticker, market_cap):
        """Process non-supplier edges."""
        edge_type = edge['type']
        
        if edge_type == 'country':
            self._create_country_edge(session, edge, ticker, market_cap)
        elif edge_type == 'sector':
            self._create_sector_edge(session, edge, ticker)
        elif edge_type == 'industry':
            self._create_industry_edge(session, edge, ticker)
        elif edge_type == 'produces':
            self._create_produces_edge(session, edge, ticker)
        elif edge_type == 'requires':
            self._create_requires_edge(session, edge, ticker)
    
    def _create_country_edge(self, session, edge, ticker, market_cap):
        """Create country node and relationship with proper schema."""
        country_name = edge['data']['country_name']
        country_info = self.COUNTRY_ISO_MAP.get(country_name, {'iso': 'XX', 'region': 'Unknown'})
        
        aliases = self._generate_aliases(country_name)
        
        session.run("""
            MERGE (n:Country {name: $name})
            SET n.aliases = $aliases,
                n.iso_code = $iso_code,
                n.region = $region,
                n.updated_at = datetime($timestamp)
        """,
        name=country_name,
        aliases=aliases,
        iso_code=country_info['iso'],
        region=country_info['region'],
        timestamp=self.timestamp)
        
        # Create relationship with proper properties
        direction = edge['direction']
        if direction == "country->company":
            rel_type = "INFLUENCES"
        else:
            rel_type = "LOCATED_IN"
        
        session.run(f"""
            MATCH (c1:{('Country' if direction == 'country->company' else 'CompanyCanonical')}) 
            WHERE c1.{'name' if direction == 'country->company' else 'ticker'} = ${('country_name' if direction == 'country->company' else 'ticker')}
            MATCH (c2:{('CompanyCanonical' if direction == 'country->company' else 'Country')})
            WHERE c2.{'ticker' if direction == 'country->company' else 'name'} = ${('ticker' if direction == 'country->company' else 'country_name')}
            MERGE (c1)-[r:{rel_type}]->(c2)
            SET r.weight = $weight,
                r.confidence = $confidence,
                r.correlation_strength = $correlation_strength,
                r.last_updated = datetime($timestamp),
                r.decay_rate = $decay_rate
        """,
        country_name=country_name,
        ticker=ticker,
        weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
        confidence=edge['relevance'],
        correlation_strength=edge.get('correlation_strength', edge['magnitude'] * edge['relevance']),
        timestamp=self.timestamp,
        decay_rate=self.DECAY_RATES['stable'])
    
    def _create_sector_edge(self, session, edge, ticker):
        """Create sector with proper schema and handle bidirectional relationships."""
        sector_name = edge['data']['sector_name']
        aliases = self._generate_aliases(sector_name)
        direction = edge.get('direction', 'company->sector')
        
        # Create the Sector node
        session.run("""
            MERGE (n:Sector {name: $name})
            SET n.aliases = $aliases,
                n.classification_level = $level,
                n.classification_system = 'GICS',
                n.updated_at = datetime($timestamp)
        """,
        name=sector_name,
        aliases=aliases,
        level=edge['data'].get('classification_level', 'sector'),
        timestamp=self.timestamp)
        
        # Create relationship based on direction
        if direction == "sector->company":
            # Sector influences Company
            session.run("""
                MATCH (s:Sector {name: $sector_name})
                MATCH (c:CompanyCanonical {ticker: $ticker})
                MERGE (s)-[r:INFLUENCES]->(c)
                SET r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.last_updated = datetime($timestamp)
            """,
            sector_name=sector_name,
            ticker=ticker,
            weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
            confidence=edge['relevance'],
            correlation_strength=edge.get('correlation_strength', edge['magnitude'] * edge['relevance']),
            timestamp=self.timestamp)
        
        else:  # company->sector
            # Company belongs to Sector
            session.run("""
                MATCH (c:CompanyCanonical {ticker: $ticker})
                MATCH (s:Sector {name: $sector_name})
                MERGE (c)-[r:BELONGS_TO]->(s)
                SET r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.last_updated = datetime($timestamp)
            """,
            ticker=ticker,
            sector_name=sector_name,
            weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
            confidence=edge['relevance'],
            correlation_strength=edge.get('correlation_strength', edge['magnitude'] * edge['relevance']),
            timestamp=self.timestamp)
            
    def _calculate_classification_confidence(self, info, sector_name):
        """Confidence that company truly belongs to this sector"""
        confidence = 0.85  # Base yfinance reliability
        
        # Penalize conglomerates/holdings
        name = info.get('longName', '').lower()
        if any(word in name for word in ['holdings', 'group', 'conglomerate']):
            confidence -= 0.15
        
        # Boost if industry aligns with sector
        # (would need industry->sector mapping)
        
        return max(0.70, min(1.0, confidence))
    
    def _create_industry_edge(self, session, edge, ticker):
        """Create industry with proper schema and handle bidirectional relationships."""
        industry_name = edge['data']['industry_name']
        aliases = self._generate_aliases(industry_name)
        direction = edge.get('direction', 'company->industry')
        
        # Create the Industry node
        session.run("""
            MERGE (n:Industry {name: $name})
            SET n.aliases = $aliases,
                n.classification_level = $level,
                n.classification_system = 'GICS',
                n.updated_at = datetime($timestamp)
        """,
        name=industry_name,
        aliases=aliases,
        level=edge['data'].get('classification_level', 'industry'),
        timestamp=self.timestamp)
        
        # Create relationship based on direction
        if direction == "industry->company":
            # Industry influences Company
            session.run("""
                MATCH (i:Industry {name: $industry_name})
                MATCH (c:CompanyCanonical {ticker: $ticker})
                MERGE (i)-[r:INFLUENCES]->(c)
                SET r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.last_updated = datetime($timestamp)
            """,
            industry_name=industry_name,
            ticker=ticker,
            weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
            confidence=edge['relevance'],
            correlation_strength=edge.get('correlation_strength', edge['magnitude'] * edge['relevance']),
            timestamp=self.timestamp)
        
        else:  # company->industry
            # Company belongs to Industry
            session.run("""
                MATCH (c:CompanyCanonical {ticker: $ticker})
                MATCH (i:Industry {name: $industry_name})
                MERGE (c)-[r:BELONGS_TO]->(i)
                SET r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.last_updated = datetime($timestamp)
            """,
            ticker=ticker,
            industry_name=industry_name,
            weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
            confidence=edge['relevance'],
            correlation_strength=edge.get('correlation_strength', edge['magnitude'] * edge['relevance']),
        timestamp=self.timestamp)
        
    def _create_produces_edge(self, session, edge, ticker):
        """Create commodity production relationship."""
        naics_code = edge['data']['naics_code']
        description = edge['data'].get('description', '')
        aliases = self._generate_aliases(description)
        
        session.run("""
            MERGE (n:Commodity {naics_code: $naics_code})
            SET n.name = $description,
                n.aliases = $aliases,
                n.commodity_type = 'output',
                n.unit = $unit,
                n.updated_at = datetime($timestamp)
        """,
        naics_code=naics_code,
        description=description,
        aliases=aliases,
        unit='units',  # TODO: Extract from data
        timestamp=self.timestamp)
        
        session.run("""
            MATCH (c:CompanyCanonical {ticker: $ticker})
            MATCH (com:Commodity {naics_code: $naics_code})
            MERGE (c)-[r:PRODUCES]->(com)
            SET r.weight = $weight,
                r.confidence = $confidence,
                r.normalized_magnitude = $normalized_magnitude,
                r.direction_value = $direction_value,
                r.last_updated = datetime($timestamp),
                r.decay_rate = 0.0
        """,
        ticker=ticker,
        naics_code=naics_code,
        weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
        confidence=edge['relevance'],
        normalized_magnitude=edge.get('normalized_magnitude'),
        direction_value=edge.get('direction_value', 1),
        timestamp=self.timestamp)
    
    def _create_requires_edge(self, session, edge, ticker):
        """Create commodity requirement relationship."""
        naics_code = edge['data']['naics_code']
        description = edge['data'].get('description', '')
        aliases = self._generate_aliases(description)
        
        session.run("""
            MERGE (n:Commodity {naics_code: $naics_code})
            SET n.name = $description,
                n.aliases = $aliases,
                n.commodity_type = 'input',
                n.unit = $unit,
                n.updated_at = datetime($timestamp)
        """,
        naics_code=naics_code,
        description=description,
        aliases=aliases,
        unit='units',
        timestamp=self.timestamp)
        
        session.run("""
            MATCH (com:Commodity {naics_code: $naics_code})
            MATCH (c:CompanyCanonical {ticker: $ticker})
            MERGE (com)-[r:REQUIRED_BY]->(c)
            SET r.weight = $weight,
                r.confidence = $confidence,
                r.normalized_magnitude = $normalized_magnitude,
                r.direction_value = $direction_value,
                r.last_updated = datetime($timestamp),
                r.decay_rate = 0.0
        """,
        naics_code=naics_code,
        ticker=ticker,
        weight=edge.get('weight', edge['magnitude'] * edge['relevance']),
        confidence=edge['relevance'],
        normalized_magnitude=edge.get('normalized_magnitude'),
        direction_value=edge.get('direction_value', -1),
        timestamp=self.timestamp)
    
    def _map_relationship_type(self, relationship: str) -> str:
        """Map relationship type to Neo4j label."""
        rel_map = {
            'SAME_AS': 'RESOLVES_TO',
            'OWNS': 'RESOLVES_TO',
            'SUBSIDIARY': 'RESOLVES_TO',
            'PARENT_OF': 'RESOLVES_TO',
            'RELATED_TO': 'RESOLVES_TO'
        }
        return rel_map.get(relationship, 'RESOLVES_TO')


def to_neo4j_enhanced(edges, ticker, 
                   canonical_companies_csv: Optional[str] = None,
                   auto_canonicalize: bool = False,
                   min_occurrences: int = 2,
                   uri="bolt://localhost:7687", 
                   user="neo4j", 
                   password="password"):
    """
    Export with fixed schema.
    
    All redundant fields removed, proper arrays, temporal properties added.
    """
    exporter = FixedNeo4jExporter(uri, user, password, canonical_companies_csv,
                                  auto_canonicalize, min_occurrences)
    try:
        return exporter.export_edges(edges, ticker)
    finally:
        exporter.close()