"""
Enhanced Neo4j Export Module
=============================

Exports graph data with all properties needed for signal propagation.

PLACEHOLDERS:
- [PH4] confidence: Uses relevance as proxy (needs NLP confidence)
- [PH5] correlation_strength: Rule-based (needs time-series analysis)
- [PH2] base_weight: Market cap proxy (needs comprehensive calculation)
"""

from neo4j import GraphDatabase
import yfinance as yf
from datetime import datetime
import math

class EnhancedNeo4jExporter:
    """
    Enhanced exporter with propagation-ready properties.
    """
    
    # Temporal classification
    TEMPORAL_TYPES = {
        'country': 'stable',
        'sector': 'stable',
        'industry': 'stable',
        'supplier': 'permanent',
        'produces': 'permanent',
        'requires': 'permanent'
    }
    
    # Decay rates (per day)
    DECAY_RATES = {
        'permanent': 0.0,
        'stable': 0.001,
        'semi-permanent': 0.05,
        'temporary': 0.15
    }
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.timestamp = datetime.now().isoformat()
    
    def close(self):
        self.driver.close()
    
    def export_edges(self, edges, ticker):
        """
        Export edges with enhanced properties.
        
        New properties added:
        - weight: Combined propagation coefficient
        - directionality: positive/negative/neutral
        - confidence: NLP confidence (uses relevance as proxy)
        - decay_rate: Temporal decay speed
        - temporal_type: permanent/stable/temporary
        - last_updated: Timestamp
        - correlation_strength: Numeric correlation
        """
        with self.driver.session() as session:
            # Get company info
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
            market_cap = info.get("marketCap", 0)
            
            # Calculate company properties
            base_weight = self._calculate_base_weight(market_cap)
            volatility_score = info.get("beta", 1.0) if info.get("beta") else 1.0
            
            # Create central company node with propagation properties
            session.run("""
                MERGE (c:Company {ticker: $ticker})
                SET c.name = $name,
                    c.marketCap = $marketCap,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.risk_signal = 0.0,
                    c.updated_at = datetime($timestamp),
                    c.base_weight = $base_weight,
                    c.volatility_score = $volatility_score,
                    c.entity_type = 'Company'
            """, 
            ticker=ticker,
            name=company_name,
            marketCap=market_cap,
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            timestamp=self.timestamp,
            base_weight=base_weight,
            volatility_score=volatility_score)
            
            # Process each edge
            for edge in edges:
                self._process_edge(session, edge, ticker, market_cap)
            
            print(f"✅ Successfully exported {len(edges)} edges with enhanced properties")
            return True
    
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
        
        # [PH4] confidence: Using relevance as proxy
        confidence = relevance
        
        # Calculate weight as combined propagation coefficient
        weight = magnitude * relevance
        
        # [PH5] correlation_strength: Rule-based conversion
        correlation_strength = self._calculate_correlation_strength(correlation, edge_type)
        
        # Directionality based on edge type and correlation
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
        
        elif edge_type == 'supplier':
            self._create_supplier_edge(session, data, ticker, direction,
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
        # Create node with propagation properties
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
                MATCH (company:Company {ticker: $ticker})
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
                MATCH (company:Company {ticker: $ticker})
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
                MATCH (company:Company {ticker: $ticker})
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
                MATCH (company:Company {ticker: $ticker})
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
        
        # Industry competition typically has negative correlation
        directionality = 'negative'
        
        if direction == "industry->company":
            session.run("""
                MATCH (industry:Industry {name: $industry_name})
                MATCH (company:Company {ticker: $ticker})
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
    
    def _create_supplier_edge(self, session, data, ticker, direction, weight,
                             confidence, correlation_strength, directionality,
                             decay_rate, temporal_type, magnitude, relevance):
        """Create supplier relationship with enhanced properties."""
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
        else:
            session.run("""
                MATCH (company:Company {ticker: $ticker})
                MATCH (supplier:Supplier {name: $supplier_name})
                MERGE (company)-[r:DEMANDS_FROM]->(supplier)
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
            ticker=ticker,
            supplier_name=data['supplier_name'],
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
            MATCH (company:Company {ticker: $ticker})
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
            MATCH (company:Company {ticker: $ticker})
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
        """
        [PH2] PLACEHOLDER: Should use revenue, employees, assets
        Current: Log-scale from market cap
        """
        if market_cap <= 0:
            return 0.5
        base_weight = 0.5 + (math.log10(market_cap) - 9) * 0.2
        return max(0.1, min(2.0, base_weight))
    
    def _calculate_correlation_strength(self, correlation, edge_type):
        """
        [PH5] PLACEHOLDER: Should analyze time-series correlations
        Current: Rule-based conversion
        """
        if edge_type == 'industry':
            return -0.7  # Competition is negative
        
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


def to_neo4j_enhanced(edges, ticker, uri="bolt://localhost:7687", 
                     user="neo4j", password="password"):
    """
    Enhanced export function with all propagation properties.
    
    Usage:
        to_neo4j_enhanced(edges, "TSLA", password="your_password")
    """
    exporter = EnhancedNeo4jExporter(uri, user, password)
    try:
        return exporter.export_edges(edges, ticker)
    finally:
        exporter.close()


if __name__ == "__main__":
    # Example: Replace existing to_neo4j call with enhanced version
    from CEG.graph.static_edges import StaticEdges
    
    static_edges = StaticEdges("BA")
    edges = static_edges.edges()
    
    to_neo4j_enhanced(
        edges, 
        "BA",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )