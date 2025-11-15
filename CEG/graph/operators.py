"""
Simple Graph Manager for Neo4j Knowledge Graph
==============================================

Two main functions:
1. add_ticker(ticker) - Adds company and all its relationships to Neo4j
2. propagate(ticker, signal) - Propagates risk signal through the graph

Usage:
    from graph_manager import add_ticker, propagate
    
    # Add a company to the database
    add_ticker("TSLA")
    
    # Propagate a negative signal (e.g., supply chain disruption)
    results = propagate("TSLA", signal=-0.7, max_hops=3)
"""

from neo4j import GraphDatabase
from datetime import datetime
import math
from collections import defaultdict

# Import your existing modules
from .static_edges import StaticEdges
from .neo4j_exporter import to_neo4j_enhanced


class GraphManager:
    """Manages Neo4j graph operations"""
    
    def __init__(self, uri="bolt://127.0.0.1:7687", user="neo4j", password="myhome2911!"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def add_ticker(self, ticker):
        """
        Add a company and all its relationships to Neo4j.
        
        Args:
            ticker: Company ticker symbol (e.g., "TSLA", "BA")
        
        Returns:
            bool: True if successful
        """
        print(f"\n{'='*80}")
        print(f"ADDING {ticker} TO NEO4J GRAPH")
        print(f"{'='*80}\n")
        
        try:
            # Generate all edges for this ticker
            print("Step 1: Generating edges (suppliers, commodities, sectors, etc.)...")
            static_edges = StaticEdges(ticker)
            edges = static_edges.edges()
            
            print(f"✅ Generated {len(edges)} edges")
            
            # Export to Neo4j with enhanced properties
            print("\nStep 2: Exporting to Neo4j with propagation properties...")
            success = to_neo4j_enhanced(
                edges, 
                ticker,
                uri=self.uri,
                user=self.user,
                password=self.password
            )
            
            if success:
                print(f"\n{'='*80}")
                print(f"✅ {ticker} SUCCESSFULLY ADDED TO GRAPH")
                print(f"{'='*80}\n")
                return True
            else:
                print(f"\n❌ Failed to export {ticker}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error adding {ticker}: {str(e)}")
            return False
    
    def propagate(self, source_ticker, signal, max_hops=3, confidence_threshold=0.5):
        """
        Propagate a risk signal from a source company through the graph.
        
        Args:
            source_ticker: Starting company ticker (e.g., "TSLA")
            signal: Signal strength (-1 to 1, negative = risk, positive = opportunity)
            max_hops: Maximum propagation distance (default: 3)
            confidence_threshold: Minimum edge confidence to propagate (default: 0.5)
        
        Returns:
            dict: Results with affected nodes and summary
        
        Example:
            # Simulate a negative event (factory fire)
            results = propagate("TSLA", signal=-0.8, max_hops=3)
            
            # Simulate a positive event (major contract win)
            results = propagate("BA", signal=0.6, max_hops=3)
        """
        print(f"\n{'='*80}")
        print(f"SIGNAL PROPAGATION: {source_ticker}")
        print(f"{'='*80}\n")
        print(f"Initial Signal: {signal}")
        print(f"Max Hops: {max_hops}")
        print(f"Confidence Threshold: {confidence_threshold}\n")
        
        with self.driver.session() as session:
            # Reset all signals first
            print("Resetting previous signals...")
            session.run("MATCH (n) WHERE n.risk_signal IS NOT NULL SET n.risk_signal = 0.0")
            
            # Initialize source signal
            session.run("""
                MATCH (c:CompanyCanonical {ticker: $ticker})
                SET c.risk_signal = $signal,
                    c.updated_at = datetime($timestamp)
            """,
            ticker=source_ticker,
            signal=signal,
            timestamp=datetime.now().isoformat())
            
            print(f"✅ Initialized {source_ticker} with signal {signal}\n")
            
            # Propagate through graph
            all_affected = {}
            current_time = datetime.now()
            
            for hop in range(max_hops):
                print(f"{'='*60}")
                print(f"HOP {hop + 1}/{max_hops}")
                print(f"{'='*60}\n")
                
                hop_results = self._propagate_one_hop(
                    session, 
                    confidence_threshold,
                    current_time
                )
                
                if not hop_results:
                    print(f"No more nodes to propagate to. Stopping at hop {hop + 1}.\n")
                    break
                
                all_affected.update(hop_results)
                print(f"✅ Affected {len(hop_results)} nodes in this hop\n")
            
            # Get summary statistics
            summary = self._get_summary(session, source_ticker)
            
            print(f"{'='*80}")
            print(f"PROPAGATION COMPLETE")
            print(f"{'='*80}\n")
            print(f"Total Affected Nodes: {summary['total_affected']}")
            print(f"\nTop 10 Most Affected:")
            for i, node in enumerate(summary['most_affected'][:10], 1):
                print(f"  {i}. {node['name']} ({node['type']}): {node['signal']:.4f}")
            
            if summary['feedback_loops']:
                print(f"\n⚠️  Detected {len(summary['feedback_loops'])} feedback loops")
            
            return {
                'affected_nodes': all_affected,
                'summary': summary
            }
    
    def _propagate_one_hop(self, session, confidence_threshold, current_time):
        """Propagate signals one hop from all active nodes"""
        
        # Get all nodes with non-zero signals and their outgoing edges
        query = """
        MATCH (source)-[r]->(target)
        WHERE source.risk_signal IS NOT NULL 
          AND source.risk_signal <> 0
          AND r.confidence >= $threshold
        RETURN 
            id(source) as source_id,
            labels(source)[0] as source_type,
            COALESCE(source.name, source.ticker) as source_name,
            source.risk_signal as source_signal,
            COALESCE(source.base_weight, 1.0) as source_weight,
            id(target) as target_id,
            labels(target)[0] as target_type,
            COALESCE(target.name, target.ticker) as target_name,
            COALESCE(target.volatility_score, 1.0) as target_volatility,
            r.weight as edge_weight,
            r.directionality as directionality,
            COALESCE(r.decay_rate, 0.01) as decay_rate,
            r.last_updated as last_updated,
            type(r) as rel_type
        """
        
        results = session.run(query, threshold=confidence_threshold)
        
        # Calculate propagated signals
        target_signals = defaultdict(float)
        target_info = {}
        
        for record in results:
            source_signal = record['source_signal']
            edge_weight = record['edge_weight']
            source_weight = record['source_weight']
            target_volatility = record['target_volatility']
            directionality = record['directionality']
            
            # Apply temporal decay
            decay_factor = 1.0
            last_updated = record['last_updated']
            if last_updated:
                try:
                    if isinstance(last_updated, str):
                        last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    time_elapsed = (current_time - last_updated).total_seconds() / 86400
                    decay_factor = math.exp(-record['decay_rate'] * time_elapsed)
                except:
                    decay_factor = 1.0
            
            # Calculate propagated signal
            propagated = source_signal * edge_weight * source_weight * target_volatility * decay_factor
            
            # Apply directionality
            if directionality == 'negative':
                propagated *= -1
            elif directionality == 'neutral':
                propagated *= 0.5
            
            target_id = record['target_id']
            target_signals[target_id] += propagated
            
            if target_id not in target_info:
                target_info[target_id] = {
                    'type': record['target_type'],
                    'name': record['target_name'],
                    'volatility': target_volatility
                }
                
                print(f"  {record['source_name']} → {record['target_name']}: {propagated:+.4f}")
        
        # Update target nodes
        affected = {}
        for target_id, signal in target_signals.items():
            # Cap signal at reasonable bounds
            signal = max(-10.0, min(10.0, signal))
            
            session.run("""
                MATCH (n)
                WHERE id(n) = $node_id
                SET n.risk_signal = COALESCE(n.risk_signal, 0.0) + $signal,
                    n.updated_at = datetime($timestamp)
            """,
            node_id=target_id,
            signal=signal,
            timestamp=current_time.isoformat())
            
            affected[target_id] = {
                'signal': signal,
                'name': target_info[target_id]['name'],
                'type': target_info[target_id]['type']
            }
        
        return affected
    
    def _get_summary(self, session, source_ticker):
        """Get summary statistics after propagation"""
        
        # Get all affected nodes
        result = session.run("""
            MATCH (n)
            WHERE n.risk_signal IS NOT NULL AND n.risk_signal <> 0
            RETURN 
                id(n) as node_id,
                labels(n)[0] as type,
                COALESCE(n.name, n.ticker) as name,
                n.risk_signal as signal
            ORDER BY abs(n.risk_signal) DESC
        """)
        
        most_affected = []
        for record in result:
            most_affected.append({
                'id': record['node_id'],
                'type': record['type'],
                'name': record['name'],
                'signal': record['signal']
            })
        
        # Detect feedback loops
        loop_result = session.run("""
            MATCH path = (a)-[r1]->(b)-[r2]->(a)
            WHERE a.risk_signal IS NOT NULL 
              AND b.risk_signal IS NOT NULL
              AND abs(a.risk_signal) > 0.1
            RETURN 
                COALESCE(a.name, a.ticker) as node_a,
                COALESCE(b.name, b.ticker) as node_b,
                a.risk_signal as signal_a,
                b.risk_signal as signal_b
            LIMIT 10
        """)
        
        feedback_loops = []
        for record in loop_result:
            feedback_loops.append({
                'node_a': record['node_a'],
                'node_b': record['node_b'],
                'signal_a': record['signal_a'],
                'signal_b': record['signal_b']
            })
        
        return {
            'total_affected': len(most_affected),
            'most_affected': most_affected,
            'feedback_loops': feedback_loops
        }
    
    def reset_all_signals(self):
        """Reset all risk signals in the graph"""
        with self.driver.session() as session:
            session.run("""
                MATCH (n)
                WHERE n.risk_signal IS NOT NULL
                SET n.risk_signal = 0.0
            """)
        print("✅ All signals reset to 0")


# =============================================================================
# SIMPLE API - Just use these two functions
# =============================================================================

# Global manager instance
_manager = None

def _get_manager():
    """Get or create the global manager instance"""
    global _manager
    if _manager is None:
        _manager = GraphManager(
            uri="bolt://127.0.0.1:7687",
            user="neo4j",
            password="myhome2911!"
        )
    return _manager


def add_ticker(ticker):
    """
    Add a company and all its relationships to Neo4j.
    
    Args:
        ticker: Company ticker symbol (e.g., "TSLA", "BA")
    
    Returns:
        bool: True if successful
    
    Example:
        add_ticker("TSLA")
    """
    manager = _get_manager()
    return manager.add_ticker(ticker)


def propagate(source_ticker, signal, max_hops=3, confidence_threshold=0.5):
    """
    Propagate a risk signal through the graph.
    
    Args:
        source_ticker: Starting company ticker (e.g., "TSLA")
        signal: Signal strength (-1 to 1, negative = risk, positive = opportunity)
        max_hops: Maximum propagation distance (default: 3)
        confidence_threshold: Minimum edge confidence (default: 0.5)
    
    Returns:
        dict: Results with affected nodes and summary
    
    Example:
        # Negative event (supply chain disruption)
        propagate("TSLA", signal=-0.7)
        
        # Positive event (major contract win)
        propagate("BA", signal=0.6)
    """
    manager = _get_manager()
    return manager.propagate(source_ticker, signal, max_hops, confidence_threshold)


def reset_signals():
    """Reset all signals in the graph"""
    manager = _get_manager()
    return manager.reset_all_signals()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Add a new company to the graph
    print("\n" + "="*80)
    print("EXAMPLE 1: Adding Boeing to the graph")
    print("="*80)
    
    add_ticker("BA")
    
    # Example 2: Propagate a negative signal (supply chain disruption)
    print("\n" + "="*80)
    print("EXAMPLE 2: Simulating a supply chain disruption")
    print("="*80)
    
    results = propagate(
        source_ticker="BA",
        signal=-0.7,  # Negative signal (risk)
        max_hops=3
    )
    
    # Example 3: Add another company and propagate positive signal
    print("\n" + "="*80)
    print("EXAMPLE 3: Adding Tesla and simulating positive news")
    print("="*80)
    
    add_ticker("TSLA")
    
    results = propagate(
        source_ticker="TSLA",
        signal=0.5,  # Positive signal (opportunity)
        max_hops=2
    )