"""
Enhanced Neo4j Exporter with Auto-Canonicalization - FIXED
==================================================

FIXES:
1. ✅ All node labels restored (Country, Sector, Industry, Commodity, Supplier)
2. ✅ Proper grouping: AVIC variants → "AVIC Holdings Corporation" (not surface entity)
3. ✅ New companies merge with existing canonical nodes

Usage:
    to_neo4j_enhanced(edges, "BA", 
                     canonical_companies_csv="company_filtered.csv",
                     auto_canonicalize=True,
                     min_occurrences=2,
                     password="your_password")
"""

from neo4j import GraphDatabase
import yfinance as yf
from datetime import datetime
import math
from typing import List, Dict, Optional, Tuple
import warnings
from collections import defaultdict
import re

# Try to import entity resolver, but work without it
try:
    from .entity_resolution_pipeline import EntityResolver
    ENTITY_RESOLUTION_AVAILABLE = True
except ImportError:
    ENTITY_RESOLUTION_AVAILABLE = False
    warnings.warn("Entity resolution not available. Install sentence-transformers and rapidfuzz.")


class EnhancedNeo4jExporter:
    """
    Enhanced exporter with auto-canonicalization.
    """
    
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
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password",
                 canonical_companies_csv: Optional[str] = None,
                 auto_canonicalize: bool = False,
                 min_occurrences: int = 2):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.timestamp = datetime.now().isoformat()
        self.auto_canonicalize = auto_canonicalize
        self.min_occurrences = min_occurrences
        
        # Initialize entity resolver if available and CSV provided
        self.resolver = None
        if canonical_companies_csv and ENTITY_RESOLUTION_AVAILABLE:
            try:
                print("🔍 Initializing entity resolver...")
                self.resolver = EntityResolver(canonical_companies_csv)
                print("✅ Entity resolver ready")
            except Exception as e:
                print(f"⚠️  Entity resolution disabled: {e}")
                self.resolver = None
        elif canonical_companies_csv and not ENTITY_RESOLUTION_AVAILABLE:
            print("⚠️  Entity resolution requested but dependencies not installed")
    
    def close(self):
        self.driver.close()
    
    def _extract_root_company_name(self, supplier_name: str) -> str:
        """
        Extract the root company name from a supplier name.
        
        Examples:
            "Panasonic Japan" → "panasonic"
            "Samsung Electronics Co Ltd" → "samsung"
            "Avic International Beichen Dong China" → "avic"
            "Avic International Holding Corporat South Korea" → "avic"
            "Avic Chengfei Commercial Aircraft" → "avic"
        """
        name = supplier_name.lower()
        
        # Remove common geographic suffixes (do this FIRST before removing other parts)
        geo_patterns = [
            r'\s+(?:japan|china|korea|south korea|north korea|germany|france|taiwan|vietnam|singapore|spain|usa|uk|united states|united kingdom)$',
            r'\s+(?:north america|south america|europe|asia|asia pacific|middle east)$',
        ]
        for pattern in geo_patterns:
            name = re.sub(pattern, '', name)
        
        # Remove common facility/division/subsidiary suffixes
        facility_patterns = [
            r'\s+(?:facility|factory|plant|division|dept|department|unit|center|branch|office).*$',
            r'\s+(?:manufacturing|industrial|commercial|international|global|regional|domestic).*$',
            r'\s+(?:aerospace|aviation|automotive|electronics|energy|technology|systems).*$',
            r'\s+(?:holding|holdings|corporat|corporation|group).*$',
        ]
        for pattern in facility_patterns:
            name = re.sub(pattern, '', name)
        
        # Remove legal suffixes
        legal_patterns = [
            r'\s+(?:inc|corp|ltd|llc|co|company|limited|plc|gmbh|ag|sa)\.?$',
        ]
        for pattern in legal_patterns:
            name = re.sub(pattern, '', name)
        
        # Clean up punctuation and extra spaces
        name = re.sub(r'[,\.\(\)]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # For multi-word names, return ONLY the first word (the actual company name)
        # This ensures "AVIC International", "AVIC Chengfei", "AVIC Beichen" all become "avic"
        words = name.split()
        if words:
            return words[0]  # Just the first word - the root company name
        
        return name
    
    def _group_suppliers_by_root_name(self, suppliers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group suppliers by their root company name.
        
        Returns:
            {
                "panasonic": [
                    {"supplier_name": "Panasonic Japan", ...},
                    {"supplier_name": "Panasonic Korea", ...}
                ],
                "avic": [
                    {"supplier_name": "Avic International Beichen Dong China", ...},
                    {"supplier_name": "Avic Chengfei Commercial Aircraft", ...}
                ]
            }
        """
        groups = defaultdict(list)
        
        for supplier in suppliers:
            root_name = self._extract_root_company_name(supplier['supplier_name'])
            groups[root_name].append(supplier)
        
        return dict(groups)
    
    def _create_auto_canonical_node(self, session, root_name: str, suppliers: List[Dict]) -> str:
        """
        Create an auto-generated canonical company node.
        
        Returns:
            Generated ticker (e.g., "AVIC_AUTO")
        """
        # Generate a pseudo-ticker from the root name
        ticker = f"{root_name.upper().replace(' ', '_')}_AUTO"
        
        # Create a proper canonical name based on the root
        # If root is "avic", canonical name is "AVIC Holdings Corporation"
        # If root is "panasonic", canonical name is "Panasonic Corporation"
        canonical_name = f"{root_name.title()} Holdings Corporation"
        
        # Extract most common location from the group
        locations = [s.get('location', '') for s in suppliers if s.get('location')]
        common_location = max(set(locations), key=locations.count) if locations else "Unknown"
        
        # Get all unique facility names for metadata
        facility_names = [s['supplier_name'] for s in suppliers]
        
        session.run("""
            MERGE (c:CompanyCanonical {ticker: $ticker})
            SET c.name = $name,
                c.root_name = $root_name,
                c.entity_type = 'CompanyCanonical',
                c.auto_generated = true,
                c.num_facilities = $num_facilities,
                c.primary_location = $location,
                c.facility_list = $facility_list,
                c.updated_at = datetime($timestamp),
                c.risk_signal = 0.0,
                c.base_weight = 0.7,
                c.volatility_score = 0.8
        """,
        ticker=ticker,
        name=canonical_name,
        root_name=root_name,
        num_facilities=len(suppliers),
        location=common_location,
        facility_list=', '.join(facility_names[:5]),  # Store first 5 for reference
        timestamp=self.timestamp)
        
        print(f"      ✨ Created: {canonical_name} ({ticker}) with {len(suppliers)} facilities")
        
        return ticker
    
    def export_edges(self, edges, ticker):
        """
        Export edges with enhanced properties and optional entity resolution.
        """
        with self.driver.session() as session:
            # Get company info
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
            market_cap = info.get("marketCap", 0)
            
            # Calculate company properties
            base_weight = self._calculate_base_weight(market_cap)
            volatility_score = info.get("beta", 1.0) if info.get("beta") else 1.0
            
            # Create central company node (CANONICAL if using resolution)
            node_label = "CompanyCanonical" if self.resolver else "Company"
            
            session.run(f"""
                MERGE (c:{node_label} {{ticker: $ticker}})
                SET c.name = $name,
                    c.legal_name = $name,
                    c.marketCap = $marketCap,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.country = $country,
                    c.risk_signal = 0.0,
                    c.updated_at = datetime($timestamp),
                    c.base_weight = $base_weight,
                    c.volatility_score = $volatility_score,
                    c.entity_type = '{node_label}',
                    c.auto_generated = false
            """, 
            ticker=ticker,
            name=company_name,
            marketCap=market_cap,
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            country=info.get("country", "Unknown"),
            timestamp=self.timestamp,
            base_weight=base_weight,
            volatility_score=volatility_score)
            
            # Separate supplier edges from others
            supplier_edges = [e for e in edges if e['type'] == 'supplier']
            other_edges = [e for e in edges if e['type'] != 'supplier']
            
            # Process suppliers
            if supplier_edges:
                if self.resolver:
                    self._process_suppliers_with_resolution(session, supplier_edges, ticker)
                else:
                    self._process_suppliers_legacy(session, supplier_edges, ticker, market_cap)
            
            # Process other edges (Country, Sector, Industry, Commodity)
            for edge in other_edges:
                self._process_edge(session, edge, ticker, market_cap)
            
            print(f"✅ Successfully exported {len(edges)} edges")
            if self.resolver:
                print(f"   - {len(supplier_edges)} supplier edges (with entity resolution)")
            else:
                print(f"   - {len(supplier_edges)} supplier edges (legacy mode)")
            print(f"   - {len(other_edges)} other edges (Country, Sector, Industry, Commodity)")
            
            return True
    
    def _process_suppliers_with_resolution(self, session, supplier_edges, ticker):
        """
        Process suppliers with entity resolution AND auto-canonicalization.
        """
        # Extract supplier data
        suppliers = []
        for edge in supplier_edges:
            suppliers.append({
                'supplier_name': edge['data']['supplier_name'],
                'location': edge['data'].get('location', ''),
                'magnitude': edge['magnitude'],
                'relevance': edge['relevance'],
                'direction': edge['direction'],
                'full_data': edge['data']
            })
        
        # Step 1: Try to resolve to companies in CSV
        print(f"   🔍 Resolving {len(suppliers)} suppliers to canonical companies...")
        resolved_suppliers = self.resolver.resolve_suppliers_batch(
            [{'supplier_name': s['supplier_name'], 'location': s['location']} 
             for s in suppliers],
            threshold=0.70
        )
        
        # Merge resolution results
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
        
        print(f"   ✅ Resolved {matched_count}/{len(suppliers)} suppliers to CSV companies")
        
        # Step 2: Auto-canonicalize unmatched suppliers if enabled
        auto_created_count = 0
        if self.auto_canonicalize and unmatched_suppliers:
            print(f"   🤖 Auto-canonicalizing {len(unmatched_suppliers)} unmatched suppliers...")
            
            # Group by root name
            groups = self._group_suppliers_by_root_name(unmatched_suppliers)
            
            # Create canonical nodes for groups with multiple suppliers
            for root_name, group_suppliers in groups.items():
                if len(group_suppliers) >= self.min_occurrences:
                    # Check if canonical node already exists from previous runs
                    auto_ticker = f"{root_name.upper().replace(' ', '_')}_AUTO"
                    
                    # Try to find existing canonical node
                    result = session.run("""
                        MATCH (c:CompanyCanonical {ticker: $ticker})
                        RETURN c.ticker as ticker
                    """, ticker=auto_ticker).single()
                    
                    if not result:
                        # Create new auto-canonical node
                        auto_ticker = self._create_auto_canonical_node(session, root_name, group_suppliers)
                        auto_created_count += 1
                    else:
                        # Node already exists, just update metadata
                        canonical_name = f"{root_name.title()} Holdings Corporation"
                        session.run("""
                            MATCH (c:CompanyCanonical {ticker: $ticker})
                            SET c.num_facilities = c.num_facilities + $new_facilities,
                                c.updated_at = datetime($timestamp)
                        """, 
                        ticker=auto_ticker,
                        new_facilities=len(group_suppliers),
                        timestamp=self.timestamp)
                        print(f"      ♻️  Merged into existing: {canonical_name} ({auto_ticker})")
                    
                    # Update all suppliers in this group
                    for supplier in group_suppliers:
                        supplier['canonical_ticker'] = auto_ticker
                        supplier['canonical_name'] = f"{root_name.title()} Holdings Corporation"
                        supplier['confidence'] = 0.95
                        supplier['match_type'] = 'auto_canonical'
                        supplier['relationship'] = 'SUBSIDIARY'
            
            if auto_created_count > 0:
                print(f"   ✅ Auto-created {auto_created_count} canonical nodes")
        
        # Step 3: Create all nodes and relationships
        for supplier in suppliers:
            # Create CompanySurface node
            session.run("""
                MERGE (s:CompanySurface {name: $name})
                SET s.location = $location,
                    s.supplier_url = $url,
                    s.product_description = $product_desc,
                    s.hs_codes = $hs_codes,
                    s.total_shipments = $shipments,
                    s.risk_signal = 0.0,
                    s.updated_at = datetime($timestamp),
                    s.entity_type = 'CompanySurface',
                    s.has_canonical_match = $has_match,
                    s.resolution_confidence = $confidence,
                    s.resolution_type = $match_type,
                    s.base_weight = 0.5,
                    s.volatility_score = 0.8
            """,
            name=supplier['supplier_name'],
            location=supplier.get('location', ''),
            url=supplier['full_data'].get('supplier_url', ''),
            product_desc=supplier['full_data'].get('product_description', ''),
            hs_codes=supplier['full_data'].get('hs_codes', ''),
            shipments=supplier['full_data'].get('total_shipments', ''),
            timestamp=self.timestamp,
            has_match=supplier['canonical_ticker'] is not None,
            confidence=supplier.get('confidence', 0.0),
            match_type=supplier.get('match_type', 'no_match'))
            
            # If resolved (from CSV or auto-created), create canonical relationship
            if supplier['canonical_ticker']:
                rel_type = self._map_relationship_type(supplier.get('relationship', 'RELATED_TO'))
                
                # Canonical node already exists (created earlier or in CSV)
                # Just create the relationship
                session.run(f"""
                    MATCH (c:CompanyCanonical {{ticker: $ticker}})
                    MATCH (s:CompanySurface {{name: $surface_name}})
                    MERGE (c)-[r:{rel_type}]->(s)
                    SET r.confidence = $confidence,
                        r.match_type = $match_type,
                        r.created_at = datetime($timestamp)
                """,
                ticker=supplier['canonical_ticker'],
                surface_name=supplier['supplier_name'],
                confidence=supplier.get('confidence', 0.0),
                match_type=supplier.get('match_type', 'unknown'),
                timestamp=self.timestamp)
            
            # Create SUPPLIES relationship
            self._create_supply_relationship_resolved(session, supplier, ticker)
        
        print(f"   📊 Final stats:")
        print(f"      - CSV matches: {matched_count}")
        print(f"      - Auto-canonicalized: {auto_created_count} groups")
        print(f"      - Standalone: {len(unmatched_suppliers) - sum(1 for s in suppliers if s.get('match_type') == 'auto_canonical')}")
    
    def _process_suppliers_legacy(self, session, supplier_edges, ticker, market_cap):
        """Legacy supplier processing (no entity resolution)."""
        for edge in supplier_edges:
            data = edge['data']
            direction = edge['direction']
            magnitude = edge['magnitude']
            relevance = edge['relevance']
            
            temporal_type = self.TEMPORAL_TYPES.get('supplier', 'permanent')
            decay_rate = self.DECAY_RATES.get(temporal_type, 0.0)
            confidence = relevance
            weight = magnitude * relevance
            correlation_strength = 0.7
            directionality = 'positive'
            
            session.run("""
                MERGE (n:Supplier {name: $supplier_name})
                SET n.url = $supplier_url,
                    n.location = $location,
                    n.product_description = $product_description,
                    n.hs_codes = $hs_codes,
                    n.risk_signal = 0.0,
                    n.updated_at = datetime($timestamp),
                    n.base_weight = 0.5,
                    n.volatility_score = 0.8,
                    n.entity_type = 'Supplier'
            """,
            supplier_name=data['supplier_name'],
            supplier_url=data.get('supplier_url', ''),
            location=data.get('location', ''),
            product_description=data.get('product_description', ''),
            hs_codes=data.get('hs_codes', ''),
            timestamp=self.timestamp)
            
            if direction == "supplier->company":
                session.run("""
                    MATCH (supplier:Supplier {name: $supplier_name})
                    MATCH (company:Company {ticker: $ticker})
                    MERGE (supplier)-[r:SUPPLIES]->(company)
                    SET r.magnitude = $magnitude,
                        r.relevance = $relevance,
                        r.weight = $weight,
                        r.confidence = $confidence,
                        r.correlation_strength = $correlation_strength,
                        r.directionality = $directionality,
                        r.decay_rate = $decay_rate,
                        r.temporal_type = $temporal_type,
                        r.last_updated = datetime($timestamp),
                        r.total_shipments = $total_shipments,
                        r.type = 'supplier'
                """,
                supplier_name=data['supplier_name'],
                ticker=ticker,
                magnitude=magnitude,
                relevance=relevance,
                weight=weight,
                confidence=confidence,
                correlation_strength=correlation_strength,
                directionality=directionality,
                decay_rate=decay_rate,
                temporal_type=temporal_type,
                timestamp=self.timestamp,
                total_shipments=data.get('total_shipments', ''))
    
    def _map_relationship_type(self, relationship: str) -> str:
        rel_map = {
            'SAME_AS': 'SAME_AS',
            'OWNS': 'OWNS',
            'SUBSIDIARY': 'SUBSIDIARY',
            'PARENT_OF': 'PARENT_OF',
            'RELATED_TO': 'RELATED_TO'
        }
        return rel_map.get(relationship, 'RELATED_TO')
    
    def _create_supply_relationship_resolved(self, session, supplier, target_ticker):
        temporal_type = 'permanent'
        decay_rate = 0.0
        confidence = supplier['relevance']
        weight = supplier['magnitude'] * supplier['relevance']
        correlation_strength = 0.7
        directionality = 'positive'
        
        session.run("""
            MATCH (s:CompanySurface {name: $supplier_name})
            MATCH (c:CompanyCanonical {ticker: $ticker})
            MERGE (s)-[r:SUPPLIES]->(c)
            SET r.magnitude = $magnitude,
                r.relevance = $relevance,
                r.weight = $weight,
                r.confidence = $confidence,
                r.correlation_strength = $correlation_strength,
                r.directionality = $directionality,
                r.decay_rate = $decay_rate,
                r.temporal_type = $temporal_type,
                r.last_updated = datetime($timestamp),
                r.total_shipments = $shipments,
                r.type = 'supplier'
        """,
        supplier_name=supplier['supplier_name'],
        ticker=target_ticker,
        magnitude=supplier['magnitude'],
        relevance=supplier['relevance'],
        weight=weight,
        confidence=confidence,
        correlation_strength=correlation_strength,
        directionality=directionality,
        decay_rate=decay_rate,
        temporal_type=temporal_type,
        timestamp=self.timestamp,
        shipments=supplier['full_data'].get('total_shipments', ''))
    
    def _process_edge(self, session, edge, ticker, market_cap):
        """Process a single edge with all properties."""
        edge_type = edge['type']
        direction = edge['direction']
        magnitude = edge['magnitude']
        relevance = edge['relevance']
        correlation = edge.get('correlation', 'positive')
        data = edge['data']
        
        # Calculate enhanced properties
        temporal_type = self.TEMPORAL_TYPES.get(edge_type, 'stable')
        decay_rate = self.DECAY_RATES.get(temporal_type, 0.01)
        confidence = relevance
        weight = magnitude * relevance
        correlation_strength = self._calculate_correlation_strength(correlation, edge_type)
        directionality = self._determine_directionality(edge_type, correlation)
        
        # Route to appropriate handler
        if edge_type == 'country':
            self._create_country_edge(session, data, ticker, direction, 
                                     weight, confidence, correlation_strength,
                                     directionality, decay_rate, temporal_type, magnitude, relevance)
        
        elif edge_type == 'sector':
            self._create_sector_edge(session, data, ticker, direction,
                                     weight, confidence, correlation_strength,
                                     directionality, decay_rate, temporal_type, magnitude, relevance)
        
        elif edge_type == 'industry':
            self._create_industry_edge(session, data, ticker, direction,
                                       weight, confidence, correlation_strength,
                                       directionality, decay_rate, temporal_type, magnitude, relevance)
        
        elif edge_type == 'produces':
            self._create_produces_edge(session, data, ticker,
                                       weight, confidence, correlation_strength,
                                       directionality, decay_rate, temporal_type, magnitude, relevance)
        
        elif edge_type == 'requires':
            self._create_requires_edge(session, data, ticker,
                                       weight, confidence, correlation_strength,
                                       directionality, decay_rate, temporal_type, magnitude, relevance)
    
    def _create_country_edge(self, session, data, ticker, direction, weight, 
                            confidence, correlation_strength, directionality,
                            decay_rate, temporal_type, magnitude, relevance):
        """Create country relationship with enhanced properties."""
        session.run("""
            MERGE (n:Country {name: $country_name})
            SET n.relationship = $relationship,
                n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.entity_type = 'Country'
        """, 
        country_name=data['country_name'],
        relationship=data['relationship'],
        timestamp=self.timestamp)
        
        if direction == "country->company":
            session.run("""
                MATCH (country:Country {name: $country_name})
                MATCH (company:CompanyCanonical {ticker: $ticker})
                MERGE (country)-[r:INFLUENCES]->(company)
                SET r.magnitude = $magnitude,
                    r.relevance = $relevance,
                    r.correlation = $correlation,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.directionality = $directionality,
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.impact_type = $impact_type,
                    r.type = 'country'
            """,
            country_name=data['country_name'],
            ticker=ticker,
            magnitude=magnitude,
            relevance=relevance,
            correlation=data.get('correlation', 'positive'),
            weight=weight,
            confidence=confidence,
            correlation_strength=correlation_strength,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=self.timestamp,
            impact_type=data.get('impact_type', ''))
        else:
            session.run("""
                MATCH (company:CompanyCanonical {ticker: $ticker})
                MATCH (country:Country {name: $country_name})
                MERGE (company)-[r:IMPACTS]->(country)
                SET r.magnitude = $magnitude,
                    r.relevance = $relevance,
                    r.correlation = $correlation,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.directionality = $directionality,
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.impact_type = $impact_type,
                    r.market_cap = $market_cap,
                    r.type = 'country'
            """,
            ticker=ticker,
            country_name=data['country_name'],
            magnitude=magnitude,
            relevance=relevance,
            correlation=data.get('correlation', 'positive'),
            weight=weight,
            confidence=confidence,
            correlation_strength=correlation_strength,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=self.timestamp,
            impact_type=data.get('impact_type', ''),
            market_cap=data.get('market_cap', 0))
    
    def _create_sector_edge(self, session, data, ticker, direction, weight,
                           confidence, correlation_strength, directionality,
                           decay_rate, temporal_type, magnitude, relevance):
        """Create sector relationship with enhanced properties."""
        session.run("""
            MERGE (n:Sector {name: $sector_name})
            SET n.classification_level = $classification_level,
                n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.volatility_score = 0.7,
                n.entity_type = 'Sector'
        """,
        sector_name=data['sector_name'],
        classification_level=data['classification_level'],
        timestamp=self.timestamp)
        
        if direction == "sector->company":
            session.run("""
                MATCH (sector:Sector {name: $sector_name})
                MATCH (company:CompanyCanonical {ticker: $ticker})
                MERGE (sector)-[r:INFLUENCES]->(company)
                SET r.magnitude = $magnitude,
                    r.relevance = $relevance,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.directionality = $directionality,
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.type = 'sector'
            """,
            sector_name=data['sector_name'],
            ticker=ticker,
            magnitude=magnitude,
            relevance=relevance,
            weight=weight,
            confidence=confidence,
            correlation_strength=correlation_strength,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=self.timestamp)
        else:
            session.run("""
                MATCH (company:CompanyCanonical {ticker: $ticker})
                MATCH (sector:Sector {name: $sector_name})
                MERGE (company)-[r:IMPACTS]->(sector)
                SET r.magnitude = $magnitude,
                    r.relevance = $relevance,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.directionality = $directionality,
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.type = 'sector'
            """,
            ticker=ticker,
            sector_name=data['sector_name'],
            magnitude=magnitude,
            relevance=relevance,
            weight=weight,
            confidence=confidence,
            correlation_strength=correlation_strength,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=self.timestamp)
    
    def _create_industry_edge(self, session, data, ticker, direction, weight,
                             confidence, correlation_strength, directionality,
                             decay_rate, temporal_type, magnitude, relevance):
        """Create industry relationship (competitive - negative correlation)."""
        session.run("""
            MERGE (n:Industry {name: $industry_name})
            SET n.classification_level = $classification_level,
                n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.volatility_score = 0.7,
                n.entity_type = 'Industry'
        """,
        industry_name=data['industry_name'],
        classification_level=data['classification_level'],
        timestamp=self.timestamp)
        
        directionality = 'negative'
        
        if direction == "industry->company":
            session.run("""
                MATCH (industry:Industry {name: $industry_name})
                MATCH (company:CompanyCanonical {ticker: $ticker})
                MERGE (industry)-[r:INFLUENCES]->(company)
                SET r.magnitude = $magnitude,
                    r.relevance = $relevance,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.correlation_strength = $correlation_strength,
                    r.directionality = $directionality,
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.type = 'industry'
            """,
            industry_name=data['industry_name'],
            ticker=ticker,
            magnitude=magnitude,
            relevance=relevance,
            weight=weight,
            confidence=confidence,
            correlation_strength=correlation_strength,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=self.timestamp)
    
    def _create_produces_edge(self, session, data, ticker, weight, confidence,
                             correlation_strength, directionality, decay_rate,
                             temporal_type, magnitude, relevance):
        """Create commodity production relationship."""
        session.run("""
            MERGE (n:Commodity {naics_code: $naics_code})
            SET n.description = $description,
                n.commodity_type = $commodity_type,
                n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.volatility_score = 0.6,
                n.entity_type = 'Commodity'
        """,
        naics_code=data['naics_code'],
        description=data.get('description', ''),
        commodity_type=data.get('commodity_type', ''),
        timestamp=self.timestamp)
        
        session.run("""
            MATCH (company:CompanyCanonical {ticker: $ticker})
            MATCH (commodity:Commodity {naics_code: $naics_code})
            MERGE (company)-[r:PRODUCES]->(commodity)
            SET r.magnitude = $magnitude,
                r.relevance = $relevance,
                r.weight = $weight,
                r.confidence = $confidence,
                r.correlation_strength = $correlation_strength,
                r.directionality = $directionality,
                r.decay_rate = $decay_rate,
                r.temporal_type = $temporal_type,
                r.last_updated = datetime($timestamp),
                r.type = 'produces'
        """,
        ticker=ticker,
        naics_code=data['naics_code'],
        magnitude=magnitude,
        relevance=relevance,
        weight=weight,
        confidence=confidence,
        correlation_strength=correlation_strength,
        directionality=directionality,
        decay_rate=decay_rate,
        temporal_type=temporal_type,
        timestamp=self.timestamp)
    
    def _create_requires_edge(self, session, data, ticker, weight, confidence,
                             correlation_strength, directionality, decay_rate,
                             temporal_type, magnitude, relevance):
        """Create commodity requirement relationship."""
        session.run("""
            MERGE (n:Commodity {naics_code: $naics_code})
            SET n.description = $description,
                n.commodity_type = $commodity_type,
                n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.volatility_score = 0.6,
                n.entity_type = 'Commodity'
        """,
        naics_code=data['naics_code'],
        description=data.get('description', ''),
        commodity_type=data.get('commodity_type', ''),
        timestamp=self.timestamp)
        
        session.run("""
            MATCH (commodity:Commodity {naics_code: $naics_code})
            MATCH (company:CompanyCanonical {ticker: $ticker})
            MERGE (commodity)-[r:REQUIRED_BY]->(company)
            SET r.magnitude = $magnitude,
                r.relevance = $relevance,
                r.weight = $weight,
                r.confidence = $confidence,
                r.correlation_strength = $correlation_strength,
                r.directionality = $directionality,
                r.decay_rate = $decay_rate,
                r.temporal_type = $temporal_type,
                r.last_updated = datetime($timestamp),
                r.layer = $layer,
                r.type = 'requires'
        """,
        naics_code=data['naics_code'],
        ticker=ticker,
        magnitude=magnitude,
        relevance=relevance,
        weight=weight,
        confidence=confidence,
        correlation_strength=correlation_strength,
        directionality=directionality,
        decay_rate=decay_rate,
        temporal_type=temporal_type,
        timestamp=self.timestamp,
        layer=data.get('layer', 'unknown'))
    
    def _calculate_base_weight(self, market_cap):
        """Calculate base weight from market cap."""
        if market_cap <= 0:
            return 0.5
        base_weight = 0.5 + (math.log10(market_cap) - 9) * 0.2
        return max(0.1, min(2.0, base_weight))
    
    def _calculate_correlation_strength(self, correlation, edge_type):
        """Calculate correlation strength."""
        if edge_type == 'industry':
            return -0.7
        if correlation == 'positive':
            return 0.7
        elif correlation == 'negative':
            return -0.7
        else:
            return 0.0
    
    def _determine_directionality(self, edge_type, correlation):
        """Determine signal directionality."""
        if edge_type == 'industry':
            return 'negative'
        elif correlation == 'negative':
            return 'negative'
        elif correlation == 'positive':
            return 'positive'
        else:
            return 'neutral'


def to_neo4j_enhanced(edges, ticker, 
                     canonical_companies_csv: Optional[str] = None,
                     auto_canonicalize: bool = False,
                     min_occurrences: int = 2,
                     uri="bolt://localhost:7687", 
                     user="neo4j", 
                     password="password"):
    """
    Enhanced export function with auto-canonicalization.
    
    Args:
        edges: List of edge dictionaries
        ticker: Company ticker
        canonical_companies_csv: Path to CSV with known companies (or None)
        auto_canonicalize: If True, auto-create canonical nodes for repeated suppliers
        min_occurrences: Minimum number of suppliers to trigger auto-canonicalization
        uri, user, password: Neo4j connection params
    
    Example:
        # Auto-create canonical nodes for supplier groups
        to_neo4j_enhanced(edges, "BA",
                         canonical_companies_csv="company_filtered.csv",
                         auto_canonicalize=True,
                         min_occurrences=2,
                         password="your_password")
    """
    exporter = EnhancedNeo4jExporter(uri, user, password, canonical_companies_csv,
                                     auto_canonicalize, min_occurrences)
    try:
        return exporter.export_edges(edges, ticker)
    finally:
        exporter.close()


if __name__ == "__main__":
    from static_edges import StaticEdges
    
    edges = StaticEdges("BA").edges()
    
    # Export with auto-canonicalization
    to_neo4j_enhanced(
        edges,
        "BA",
        canonical_companies_csv="companies_filtered.csv",
        auto_canonicalize=True,
        min_occurrences=2,
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )