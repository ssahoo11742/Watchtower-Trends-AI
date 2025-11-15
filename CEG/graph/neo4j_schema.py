"""
Enhanced Neo4j Schema Implementation with Signal Propagation Support
====================================================================

This module upgrades your existing Neo4j graph with the properties needed for:
- Signal propagation
- Causal inference
- Temporal weighting
- Economic contagion modeling

PLACEHOLDERS (features requiring additional data/implementation):
- [PH1] volatility_score calculation (needs historical price data)
- [PH2] base_weight calculation (needs company size/revenue data)
- [PH3] decay_rate tuning (needs historical event-response data)
- [PH4] confidence scoring (needs NLP extraction confidence)
- [PH5] correlation_strength (needs time-series analysis)
"""

from neo4j import GraphDatabase
import yfinance as yf
from datetime import datetime
import math

class EnhancedNeo4jSchema:
    """
    Enhances existing Neo4j graph with propagation-ready properties.
    """
    
    # Temporal type classification for edges
    TEMPORAL_TYPES = {
        'supplier': 'permanent',
        'produces': 'permanent',
        'requires': 'permanent',
        'country': 'stable',
        'sector': 'stable',
        'industry': 'stable',
        'event': 'temporary',
        'news': 'temporary',
        'regulation': 'semi-permanent'
    }
    
    # Default decay rates (per day)
    DECAY_RATES = {
        'permanent': 0.0,
        'stable': 0.001,
        'semi-permanent': 0.05,
        'temporary': 0.15
    }
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def upgrade_nodes(self, ticker):
        """
        Add propagation properties to all nodes.
        
        New Properties Added:
        - risk_signal: Current propagated signal value
        - updated_at: Timestamp for temporal calculations
        - base_weight: Economic inertia/influence
        - volatility_score: Sensitivity to signals
        - entity_type: Node classification
        """
        with self.driver.session() as session:
            # Upgrade Company nodes
            self._upgrade_company_nodes(session, ticker)
            
            # Upgrade Supplier nodes
            self._upgrade_supplier_nodes(session)
            
            # Upgrade Commodity nodes
            self._upgrade_commodity_nodes(session)
            
            # Upgrade Country nodes
            self._upgrade_country_nodes(session)
            
            # Upgrade Sector/Industry nodes
            self._upgrade_sector_industry_nodes(session)
    
    def _upgrade_company_nodes(self, session, ticker):
        """Upgrade Company nodes with propagation properties."""
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap", 0)
        
        # [PH2] base_weight: Should be calculated from revenue, market cap, employee count
        # Placeholder: Use market cap normalization
        base_weight = self._calculate_base_weight(market_cap)
        
        # [PH1] volatility_score: Should be calculated from historical price volatility
        # Placeholder: Use beta as proxy, or default to 1.0
        volatility_score = info.get("beta", 1.0) if info.get("beta") else 1.0
        
        session.run("""
            MATCH (c:Company {ticker: $ticker})
            SET c.risk_signal = 0.0,
                c.updated_at = datetime($timestamp),
                c.base_weight = $base_weight,
                c.volatility_score = $volatility_score,
                c.entity_type = 'Company'
        """, 
        ticker=ticker,
        timestamp=datetime.now().isoformat(),
        base_weight=base_weight,
        volatility_score=volatility_score)
        
        print(f"✅ Upgraded Company node: {ticker}")
        print(f"   base_weight: {base_weight:.2f}, volatility: {volatility_score:.2f}")
    
    def _upgrade_supplier_nodes(self, session):
        """Upgrade Supplier nodes with propagation properties."""
        
        # [PH2] Supplier base_weight: Should be calculated from shipment volume, revenue
        # Placeholder: Use normalized shipment count
        session.run("""
            MATCH (s:Supplier)
            SET s.risk_signal = 0.0,
                s.updated_at = datetime($timestamp),
                s.base_weight = 0.5,
                s.volatility_score = 0.8,
                s.entity_type = 'Supplier'
        """, timestamp=datetime.now().isoformat())
        
        print("✅ Upgraded Supplier nodes")
    
    def _upgrade_commodity_nodes(self, session):
        """Upgrade Commodity nodes with propagation properties."""
        session.run("""
            MATCH (c:Commodity)
            SET c.risk_signal = 0.0,
                c.updated_at = datetime($timestamp),
                c.volatility_score = 0.6,
                c.entity_type = 'Commodity'
        """, timestamp=datetime.now().isoformat())
        
        print("✅ Upgraded Commodity nodes")
    
    def _upgrade_country_nodes(self, session):
        """Upgrade Country nodes with propagation properties."""
        session.run("""
            MATCH (c:Country)
            SET c.risk_signal = 0.0,
                c.updated_at = datetime($timestamp),
                c.entity_type = 'Country'
        """, timestamp=datetime.now().isoformat())
        
        print("✅ Upgraded Country nodes")
    
    def _upgrade_sector_industry_nodes(self, session):
        """Upgrade Sector and Industry nodes with propagation properties."""
        session.run("""
            MATCH (n)
            WHERE n:Sector OR n:Industry
            SET n.risk_signal = 0.0,
                n.updated_at = datetime($timestamp),
                n.volatility_score = 0.7,
                n.entity_type = labels(n)[0]
        """, timestamp=datetime.now().isoformat())
        
        print("✅ Upgraded Sector/Industry nodes")
    
    def upgrade_edges(self):
        """
        Add propagation properties to all edges.
        
        New Properties Added:
        - weight: Final propagation coefficient
        - directionality: positive/negative/neutral
        - confidence: NLP extraction confidence
        - decay_rate: Temporal decay speed
        - temporal_type: permanent/stable/temporary
        - last_updated: Timestamp
        - correlation_strength: Numeric correlation value
        """
        with self.driver.session() as session:
            self._upgrade_all_edges(session)
    
    def _upgrade_all_edges(self, session):
        """Upgrade all relationship types with propagation properties."""
        
        # Get all relationship types
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record[0] for record in result]
        
        for rel_type in rel_types:
            edge_type = self._map_rel_type_to_edge_type(rel_type)
            temporal_type = self.TEMPORAL_TYPES.get(edge_type, 'stable')
            decay_rate = self.DECAY_RATES.get(temporal_type, 0.01)
            
            # Determine directionality based on relationship type
            directionality = self._determine_directionality(rel_type)
            
            # [PH4] confidence: Should come from NLP extraction
            # Placeholder: Use relevance as proxy, or default 0.85
            
            # [PH5] correlation_strength: Should be calculated from time-series
            # Placeholder: Use correlation field if exists, else default
            
            session.run(f"""
                MATCH ()-[r:{rel_type}]->()
                SET r.weight = COALESCE(r.magnitude * r.relevance, r.magnitude, 0.5),
                    r.directionality = $directionality,
                    r.confidence = COALESCE(r.relevance, 0.85),
                    r.decay_rate = $decay_rate,
                    r.temporal_type = $temporal_type,
                    r.last_updated = datetime($timestamp),
                    r.correlation_strength = CASE 
                        WHEN r.correlation = 'positive' THEN 0.7
                        WHEN r.correlation = 'negative' THEN -0.7
                        ELSE 0.0
                    END
            """,
            directionality=directionality,
            decay_rate=decay_rate,
            temporal_type=temporal_type,
            timestamp=datetime.now().isoformat())
            
            print(f"✅ Upgraded {rel_type} edges (temporal: {temporal_type}, decay: {decay_rate})")
    
    def _calculate_base_weight(self, market_cap):
        """
        Calculate base_weight from market cap.
        
        [PH2] PLACEHOLDER: Should incorporate:
        - Revenue
        - Employee count
        - Asset size
        - Industry position
        
        Current: Simple log-scale normalization
        """
        if market_cap <= 0:
            return 0.5
        
        # Log-scale normalization (1B = 0.5, 1T = 1.5)
        base_weight = 0.5 + (math.log10(market_cap) - 9) * 0.2
        return max(0.1, min(2.0, base_weight))
    
    def _map_rel_type_to_edge_type(self, rel_type):
        """Map Neo4j relationship type to edge type classification."""
        mapping = {
            'SUPPLIES': 'supplier',
            'DEMANDS_FROM': 'supplier',
            'PRODUCES': 'produces',
            'REQUIRED_BY': 'requires',
            'INFLUENCES': 'country',
            'IMPACTS': 'sector',
            'COMPETES_WITH': 'industry'
        }
        return mapping.get(rel_type, 'stable')
    
    def _determine_directionality(self, rel_type):
        """
        Determine if relationship propagates positive/negative signals.
        
        [PH5] PLACEHOLDER: Should analyze historical correlations
        Current: Rule-based classification
        """
        negative_rels = ['COMPETES_WITH', 'DISRUPTS']
        positive_rels = ['SUPPLIES', 'PRODUCES', 'PARTNERS_WITH']
        
        if rel_type in negative_rels:
            return 'negative'
        elif rel_type in positive_rels:
            return 'positive'
        else:
            return 'neutral'
    
    def initialize_signal(self, ticker, initial_signal=0.0):
        """
        Initialize risk signal for a company.
        Used to start propagation simulations.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Company {ticker: $ticker})
                SET c.risk_signal = $signal,
                    c.updated_at = datetime($timestamp)
            """,
            ticker=ticker,
            signal=initial_signal,
            timestamp=datetime.now().isoformat())
            
            print(f"✅ Initialized signal for {ticker}: {initial_signal}")
    
    def validate_schema(self):
        """
        Validate that all required properties exist.
        Returns list of missing properties.
        """
        with self.driver.session() as session:
            # Check node properties
            node_check = session.run("""
                MATCH (n)
                WHERE n:Company OR n:Supplier OR n:Commodity
                RETURN 
                    labels(n)[0] as type,
                    COUNT(*) as total,
                    COUNT(n.risk_signal) as has_signal,
                    COUNT(n.updated_at) as has_timestamp,
                    COUNT(n.base_weight) as has_weight,
                    COUNT(n.volatility_score) as has_volatility
            """)
            
            print("\n📊 Node Property Coverage:")
            for record in node_check:
                print(f"{record['type']}: {record['total']} nodes")
                print(f"  - risk_signal: {record['has_signal']}/{record['total']}")
                print(f"  - updated_at: {record['has_timestamp']}/{record['total']}")
                print(f"  - base_weight: {record['has_weight']}/{record['total']}")
                print(f"  - volatility_score: {record['has_volatility']}/{record['total']}")
            
            # Check edge properties
            edge_check = session.run("""
                MATCH ()-[r]->()
                RETURN 
                    type(r) as rel_type,
                    COUNT(*) as total,
                    COUNT(r.weight) as has_weight,
                    COUNT(r.decay_rate) as has_decay,
                    COUNT(r.confidence) as has_confidence
            """)
            
            print("\n📊 Edge Property Coverage:")
            for record in edge_check:
                print(f"{record['rel_type']}: {record['total']} edges")
                print(f"  - weight: {record['has_weight']}/{record['total']}")
                print(f"  - decay_rate: {record['has_decay']}/{record['total']}")
                print(f"  - confidence: {record['has_confidence']}/{record['total']}")


def upgrade_existing_graph(ticker, uri="bolt://localhost:7687", user="neo4j", password="password"):
    """
    Main function to upgrade existing Neo4j graph.
    
    Usage:
        upgrade_existing_graph("TSLA", password="your_password")
    """
    schema = EnhancedNeo4jSchema(uri, user, password)
    
    try:
        print("\n" + "="*80)
        print("UPGRADING NEO4J GRAPH FOR SIGNAL PROPAGATION")
        print("="*80 + "\n")
        
        print("Step 1: Upgrading Nodes...")
        schema.upgrade_nodes(ticker)
        
        print("\nStep 2: Upgrading Edges...")
        schema.upgrade_edges()
        
        print("\nStep 3: Validating Schema...")
        schema.validate_schema()
        
        print("\n" + "="*80)
        print("✅ UPGRADE COMPLETE")
        print("="*80)
        
        print("\n⚠️  PLACEHOLDERS TO ADDRESS:")
        print("  [PH1] volatility_score: Calculate from historical price data")
        print("  [PH2] base_weight: Calculate from revenue/size metrics")
        print("  [PH3] decay_rate: Tune from historical event-response data")
        print("  [PH4] confidence: Extract from NLP pipeline")
        print("  [PH5] correlation_strength: Calculate from time-series analysis")
        
    finally:
        schema.close()


if __name__ == "__main__":
    # Example usage
    upgrade_existing_graph(
        ticker="BA",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )