"""
Signal Propagation Engine for Neo4j Graph
==========================================

Implements mathematical signal propagation across the knowledge graph.
Supports:
- Weighted propagation based on edge properties
- Temporal decay
- Multi-hop contagion
- Reflexivity detection

PLACEHOLDERS:
- [PH6] Advanced reflexivity loops (needs feedback detection algorithm)
- [PH7] Bayesian confidence updating (needs prior distributions)
- [PH8] Multi-agent swarm coordination (needs agent framework)
"""

from neo4j import GraphDatabase
from datetime import datetime, timedelta
import math
from collections import defaultdict

class SignalPropagation:
    """
    Propagates risk signals through the knowledge graph.
    """
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def propagate_signal(self, source_ticker, initial_signal, max_hops=3, 
                        confidence_threshold=0.5, decay_enabled=True):
        """
        Propagate signal from source node through the graph.
        
        Args:
            source_ticker: Starting company ticker
            initial_signal: Initial signal value (-1 to 1, negative = risk)
            max_hops: Maximum propagation distance
            confidence_threshold: Minimum edge confidence to propagate
            decay_enabled: Whether to apply temporal decay
        
        Returns:
            dict: {node_id: final_signal} for all affected nodes
        """
        with self.driver.session() as session:
            # Initialize source signal
            session.run("""
                MATCH (c:Company {ticker: $ticker})
                SET c.risk_signal = $signal,
                    c.updated_at = datetime($timestamp)
            """,
            ticker=source_ticker,
            signal=initial_signal,
            timestamp=datetime.now().isoformat())
            
            print(f"\n🔥 Initializing propagation from {source_ticker} with signal {initial_signal}")
            
            # Propagate through graph
            affected_nodes = {}
            for hop in range(max_hops):
                print(f"\n{'='*60}")
                print(f"HOP {hop + 1}/{max_hops}")
                print(f"{'='*60}")
                
                hop_results = self._propagate_one_hop(
                    session, 
                    confidence_threshold,
                    decay_enabled
                )
                
                if not hop_results:
                    print(f"No more nodes to propagate to. Stopping at hop {hop + 1}.")
                    break
                
                affected_nodes.update(hop_results)
                print(f"✅ Affected {len(hop_results)} nodes in this hop")
            
            return affected_nodes
    
    def _propagate_one_hop(self, session, confidence_threshold, decay_enabled):
        """
        Propagate signals one hop from all active nodes.
        """
        current_time = datetime.now()
        
        # Get all nodes with non-zero signals
        query = """
        MATCH (source)-[r]->(target)
        WHERE source.risk_signal IS NOT NULL 
          AND source.risk_signal <> 0
          AND r.confidence >= $threshold
        RETURN 
            id(source) as source_id,
            labels(source)[0] as source_type,
            source.risk_signal as source_signal,
            source.base_weight as source_weight,
            id(target) as target_id,
            labels(target)[0] as target_type,
            target.volatility_score as target_volatility,
            r.weight as edge_weight,
            r.directionality as directionality,
            r.decay_rate as decay_rate,
            r.temporal_type as temporal_type,
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
            source_weight = record.get('source_weight', 1.0)
            target_volatility = record.get('target_volatility', 1.0)
            directionality = record['directionality']
            
            # Apply temporal decay if enabled
            decay_factor = 1.0
            if decay_enabled and record['decay_rate']:
                last_updated = record['last_updated']
                if last_updated:
                    # [PH3] PLACEHOLDER: This assumes last_updated is datetime
                    # May need to parse string if stored as ISO format
                    try:
                        if isinstance(last_updated, str):
                            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        time_elapsed = (current_time - last_updated).total_seconds() / 86400  # days
                        decay_factor = math.exp(-record['decay_rate'] * time_elapsed)
                    except:
                        decay_factor = 1.0
            
            # Calculate propagated signal
            # Formula: signal * edge_weight * source_weight * target_volatility * decay_factor
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
                    'volatility': target_volatility
                }
        
        # Update target nodes in database
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
            
            affected[target_id] = signal
            
            node_type = target_info[target_id]['type']
            print(f"  {node_type} (ID: {target_id}): signal = {signal:.4f}")
        
        return affected
    
    def get_most_affected_nodes(self, top_n=10):
        """
        Get the most affected nodes by absolute signal strength.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.risk_signal IS NOT NULL AND n.risk_signal <> 0
                RETURN 
                    id(n) as node_id,
                    labels(n)[0] as type,
                    COALESCE(n.name, n.ticker, 'Unknown') as name,
                    n.risk_signal as signal,
                    n.updated_at as updated
                ORDER BY abs(n.risk_signal) DESC
                LIMIT $top_n
            """, top_n=top_n)
            
            nodes = []
            for record in result:
                nodes.append({
                    'id': record['node_id'],
                    'type': record['type'],
                    'name': record['name'],
                    'signal': record['signal'],
                    'updated': record['updated']
                })
            
            return nodes
    
    def detect_reflexivity_loops(self, threshold=0.1):
        """
        Detect feedback loops in the graph where signals amplify.
        
        [PH6] PLACEHOLDER: Advanced loop detection needs:
        - Cycle detection algorithm
        - Feedback strength calculation
        - Damping factor computation
        
        Current: Simple circular relationship detection
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (a)-[r1]->(b)-[r2]->(a)
                WHERE a.risk_signal IS NOT NULL 
                  AND b.risk_signal IS NOT NULL
                  AND abs(a.risk_signal) > $threshold
                RETURN 
                    COALESCE(a.name, a.ticker) as node_a,
                    COALESCE(b.name, b.ticker) as node_b,
                    a.risk_signal as signal_a,
                    b.risk_signal as signal_b,
                    type(r1) as rel_1,
                    type(r2) as rel_2
            """, threshold=threshold)
            
            loops = []
            for record in result:
                loops.append({
                    'node_a': record['node_a'],
                    'node_b': record['node_b'],
                    'signal_a': record['signal_a'],
                    'signal_b': record['signal_b'],
                    'relationship_1': record['rel_1'],
                    'relationship_2': record['rel_2']
                })
            
            return loops
    
    def reset_signals(self):
        """Reset all risk signals to 0."""
        with self.driver.session() as session:
            session.run("""
                MATCH (n)
                WHERE n.risk_signal IS NOT NULL
                SET n.risk_signal = 0.0
            """)
            print("✅ All signals reset to 0")
    
    def simulate_event(self, event_description, affected_entity, signal_strength):
        """
        Simulate an external event and propagate its effects.
        
        Args:
            event_description: Description of event (e.g., "Tesla factory strike")
            affected_entity: Ticker or name of affected entity
            signal_strength: Event impact (-1 to 1)
        """
        print(f"\n{'='*80}")
        print(f"EVENT SIMULATION: {event_description}")
        print(f"{'='*80}\n")
        print(f"Affected Entity: {affected_entity}")
        print(f"Signal Strength: {signal_strength}")
        
        # Propagate signal
        affected_nodes = self.propagate_signal(
            source_ticker=affected_entity,
            initial_signal=signal_strength,
            max_hops=3
        )
        
        print(f"\n{'='*80}")
        print(f"EVENT IMPACT SUMMARY")
        print(f"{'='*80}\n")
        print(f"Total affected nodes: {len(affected_nodes)}")
        
        # Show most affected
        most_affected = self.get_most_affected_nodes(top_n=10)
        print(f"\nTop 10 Most Affected Entities:")
        for i, node in enumerate(most_affected, 1):
            print(f"{i}. {node['name']} ({node['type']}): {node['signal']:.4f}")
        
        # Check for reflexivity
        loops = self.detect_reflexivity_loops()
        if loops:
            print(f"\n⚠️  Detected {len(loops)} feedback loops:")
            for loop in loops[:5]:  # Show first 5
                print(f"  {loop['node_a']} ↔ {loop['node_b']}")
        
        return affected_nodes


# Cypher query templates for advanced operations
ADVANCED_QUERIES = {
    'community_detection': """
        // [PH8] PLACEHOLDER: Requires graph algorithms plugin
        // CALL gds.louvain.stream('myGraph')
        // YIELD nodeId, communityId
        // Current: Manual community detection not implemented
    """,
    
    'pagerank_influence': """
        // [PH8] PLACEHOLDER: Requires graph algorithms plugin
        // CALL gds.pageRank.stream('myGraph')
        // YIELD nodeId, score
        // Current: Using base_weight as proxy for influence
    """,
    
    'shortest_path_risk': """
        // Find shortest path between two entities
        MATCH path = shortestPath(
            (a:Company {ticker: $ticker_a})-[*..5]-(b:Company {ticker: $ticker_b})
        )
        RETURN path, 
               [rel in relationships(path) | rel.weight] as weights,
               reduce(s = 1.0, w in [rel in relationships(path) | rel.weight] | s * w) as total_weight
    """
}


if __name__ == "__main__":
    # Example usage
    propagator = SignalPropagation(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )
    
    try:
        # Reset previous signals
        propagator.reset_signals()
        
        # Simulate negative event (e.g., supplier disruption)
        propagator.simulate_event(
            event_description="Major supplier factory fire",
            affected_entity="BA",  # Boeing
            signal_strength=-0.7  # Negative signal
        )
        
    finally:
        propagator.close()