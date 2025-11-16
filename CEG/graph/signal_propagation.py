"""
Modern Signal Propagation Engine with Entity Tracking
=====================================================

Tracks signal changes to specific entities at each propagation hop.
"""

from neo4j import GraphDatabase
from datetime import datetime
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class ModernPropagation:
    """
    Propagates signals through knowledge graph with detailed entity tracking.
    """
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.timestamp = datetime.now().isoformat()
        self.entity_tracker = defaultdict(list)  # Track changes per entity
    
    def close(self):
        self.driver.close()
    
    def propagate_shock(self, 
                       source_ticker: str,
                       shock_magnitude: float,
                       shock_type: str = "supply_disruption",
                       max_hops: int = 3,
                       min_propagation_threshold: float = 0.01,
                       track_entities: List[str] = None,
                       feedback_mode: str = "realistic") -> Dict:
        """
        Propagate a shock through the network with entity tracking.
        
        Args:
            source_ticker: Starting company ticker
            shock_magnitude: Initial shock strength (-1.0 to 1.0)
            shock_type: Type of shock
            max_hops: Maximum propagation distance
            min_propagation_threshold: Minimum signal strength to continue
            track_entities: List of tickers/names to track at each hop
            feedback_mode: How to handle feedback loops:
                - "realistic" (default): Allow feedback with damping
                - "isolated": Prevent source from receiving any signals
                - "full": Allow unlimited feedback (can amplify unrealistically)
        
        Returns:
            Dict with propagation results and per-entity evolution
        """
        
        print(f"\n{'='*80}")
        print(f"🚨 SHOCK PROPAGATION: {shock_type}")
        print(f"{'='*80}")
        print(f"Source: {source_ticker}")
        print(f"Magnitude: {shock_magnitude:+.4f}")
        print(f"Max Hops: {max_hops}")
        print(f"Feedback Mode: {feedback_mode}")
        if track_entities:
            print(f"Tracking: {', '.join(track_entities)}")
        
        with self.driver.session() as session:
            # Reset entity tracker
            self.entity_tracker.clear()
            
            # Get source node elementId and initial signal for feedback handling
            source_element_id = None
            source_initial_signal = shock_magnitude
            result = session.run("""
                MATCH (c:CompanyCanonical {ticker: $ticker})
                RETURN elementId(c) as element_id
            """, ticker=source_ticker)
            record = result.single()
            if record:
                source_element_id = record['element_id']
            
            # Initialize source node signal
            self._initialize_signal(session, source_ticker, shock_magnitude)
            
            # Track initial state for all tracked entities
            if track_entities:
                self._snapshot_entities(session, track_entities, hop=0, phase="initial")
            
            # Track propagation results
            all_affected = {}
            hop_summary = []
            
            # Propagate through hops
            for hop in range(1, max_hops + 1):
                print(f"\n{'─'*80}")
                print(f"HOP {hop}/{max_hops}")
                print(f"{'─'*80}")
                
                # Snapshot BEFORE propagation
                if track_entities:
                    self._snapshot_entities(session, track_entities, hop=hop, phase="before")
                
                affected_nodes = self._propagate_one_hop(
                    session,
                    min_threshold=min_propagation_threshold,
                    shock_type=shock_type,
                    exclude_element_id=source_element_id if feedback_mode == "isolated" else None
                )
                
                # Apply feedback damping for realistic mode
                if feedback_mode == "realistic" and source_element_id:
                    self._apply_feedback_damping(
                        session, 
                        source_element_id, 
                        source_initial_signal,
                        damping_factor=0.005  # Only allow 30% of feedback to affect source
                    )
                
                # Snapshot AFTER propagation
                if track_entities:
                    self._snapshot_entities(session, track_entities, hop=hop, phase="after")
                    self._print_entity_changes(track_entities, hop)
                
                if not affected_nodes:
                    print(f"✓ No more nodes affected. Stopping at hop {hop}.")
                    break
                
                # Update aggregate results
                for node_id, info in affected_nodes.items():
                    if node_id not in all_affected:
                        all_affected[node_id] = info
                    else:
                        all_affected[node_id]['signal'] += info['signal']
                        all_affected[node_id]['hop'] = min(all_affected[node_id]['hop'], info['hop'])
                
                # After feedback damping, update the affected_nodes entry for source if it exists
                if feedback_mode == "realistic" and source_element_id:
                    # Get actual signal value from database after damping
                    actual_signal = session.run("""
                        MATCH (n)
                        WHERE elementId(n) = $source_id
                        RETURN n.signal as signal
                    """, source_id=source_element_id).single()
                    
                    if actual_signal and source_element_id in all_affected:
                        all_affected[source_element_id]['signal'] = actual_signal['signal']
                
                hop_summary.append({
                    'hop': hop,
                    'nodes_affected': len(affected_nodes),
                    'total_signal_magnitude': sum(abs(n['signal']) for n in affected_nodes.values())
                })
                
                print(f"  → {len(affected_nodes)} nodes affected")
            
            # Generate summary
            summary = self._generate_summary(session, all_affected, hop_summary)
            
            # Add entity tracking to results
            if track_entities:
                summary['entity_evolution'] = self._get_entity_evolution(track_entities)
            
            return {
                'source': source_ticker,
                'shock_magnitude': shock_magnitude,
                'shock_type': shock_type,
                'affected_nodes': all_affected,
                'summary': summary,
                'entity_tracking': dict(self.entity_tracker) if track_entities else {}
            }
    
    def _snapshot_entities(self, session, entity_ids: List[str], hop: int, phase: str):
        """
        Capture current signal state for tracked entities.
        
        Args:
            entity_ids: List of tickers or names to track
            hop: Current hop number
            phase: 'initial', 'before', or 'after'
        """
        query = """
        MATCH (n)
        WHERE n.ticker IN $entity_ids OR n.name IN $entity_ids
        RETURN 
            COALESCE(n.ticker, n.name) as identifier,
            labels(n)[0] as type,
            COALESCE(n.name, n.ticker) as name,
            COALESCE(n.signal, 0.0) as signal,
            elementId(n) as node_id
        """
        
        results = session.run(query, entity_ids=entity_ids)
        
        for record in results:
            identifier = record['identifier']
            self.entity_tracker[identifier].append({
                'hop': hop,
                'phase': phase,
                'signal': record['signal'],
                'name': record['name'],
                'type': record['type'],
                'node_id': record['node_id']
            })
    
    def _print_entity_changes(self, entity_ids: List[str], hop: int):
        """Print changes for tracked entities at this hop."""
        print(f"\n  📊 Tracked Entity Changes (Hop {hop}):")
        
        for entity_id in entity_ids:
            if entity_id not in self.entity_tracker:
                continue
            
            snapshots = self.entity_tracker[entity_id]
            
            # Find before and after for this hop
            before = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'before'), None)
            after = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'after'), None)
            
            if before and after:
                change = after['signal'] - before['signal']
                if abs(change) > 0.0001:  # Only show meaningful changes
                    print(f"     {entity_id:15} | {before['signal']:+.4f} → {after['signal']:+.4f} (Δ {change:+.4f})")
                else:
                    print(f"     {entity_id:15} | {before['signal']:+.4f} (no change)")
    
    def _get_entity_evolution(self, entity_ids: List[str]) -> Dict:
        """Get complete evolution summary for tracked entities."""
        evolution = {}
        
        for entity_id in entity_ids:
            if entity_id not in self.entity_tracker:
                evolution[entity_id] = {'error': 'Entity not found'}
                continue
            
            snapshots = self.entity_tracker[entity_id]
            
            # Get initial and final states
            initial = next((s for s in snapshots if s['phase'] == 'initial'), None)
            final = snapshots[-1] if snapshots else None
            
            # Calculate hop-by-hop changes
            hop_changes = []
            for hop_num in set(s['hop'] for s in snapshots if s['hop'] > 0):
                before = next((s for s in snapshots if s['hop'] == hop_num and s['phase'] == 'before'), None)
                after = next((s for s in snapshots if s['hop'] == hop_num and s['phase'] == 'after'), None)
                
                if before and after:
                    hop_changes.append({
                        'hop': hop_num,
                        'before': before['signal'],
                        'after': after['signal'],
                        'change': after['signal'] - before['signal']
                    })
            
            evolution[entity_id] = {
                'name': snapshots[0]['name'] if snapshots else entity_id,
                'type': snapshots[0]['type'] if snapshots else 'Unknown',
                'initial_signal': initial['signal'] if initial else 0.0,
                'final_signal': final['signal'] if final else 0.0,
                'total_change': (final['signal'] if final else 0.0) - (initial['signal'] if initial else 0.0),
                'hop_by_hop': hop_changes
            }
        
        return evolution
    
    def print_entity_evolution_report(self, entity_ids: List[str] = None):
        """Print detailed evolution report for tracked entities."""
        if not self.entity_tracker:
            print("No entities tracked in last propagation.")
            return
        
        entities_to_report = entity_ids if entity_ids else list(self.entity_tracker.keys())
        
        print(f"\n{'='*80}")
        print(f"ENTITY EVOLUTION REPORT")
        print(f"{'='*80}")
        
        for entity_id in entities_to_report:
            if entity_id not in self.entity_tracker:
                print(f"\n❌ {entity_id}: Not found")
                continue
            
            snapshots = self.entity_tracker[entity_id]
            entity_name = snapshots[0]['name']
            entity_type = snapshots[0]['type']
            
            print(f"\n📈 {entity_name} ({entity_type})")
            print(f"{'─'*80}")
            
            # Group by hop
            hops = sorted(set(s['hop'] for s in snapshots))
            
            for hop in hops:
                hop_snapshots = [s for s in snapshots if s['hop'] == hop]
                
                if hop == 0:
                    initial = hop_snapshots[0]
                    print(f"  Hop 0 (Initial): {initial['signal']:+.4f}")
                else:
                    before = next((s for s in hop_snapshots if s['phase'] == 'before'), None)
                    after = next((s for s in hop_snapshots if s['phase'] == 'after'), None)
                    
                    if before and after:
                        change = after['signal'] - before['signal']
                        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                        print(f"  Hop {hop}: {before['signal']:+.4f} → {after['signal']:+.4f}  {arrow} {change:+.4f}")
            
            # Summary
            initial_val = snapshots[0]['signal']
            final_val = snapshots[-1]['signal']
            total_change = final_val - initial_val
            
            print(f"\n  Summary:")
            print(f"    Initial:       {initial_val:+.4f}")
            print(f"    Final:         {final_val:+.4f}")
            print(f"    Total Change:  {total_change:+.4f}")
    
    def _apply_feedback_damping(self, session, source_element_id: str, 
                               initial_signal: float, damping_factor: float = 0.3):
        """
        Apply damping to feedback received by source node.
        
        In realistic mode, the source can receive feedback, but it's dampened
        to prevent unrealistic amplification. This models:
        - Market adaptation and resilience
        - Management response to cascading effects
        - Natural friction in economic systems
        """
        result = session.run("""
            MATCH (n)
            WHERE elementId(n) = $source_id
            WITH n, n.signal as current_signal, $initial_signal as initial_signal
            // Calculate feedback component (signal beyond initial shock)
            WITH n, current_signal, initial_signal,
                 current_signal - initial_signal as feedback
            // Apply damping to feedback only
            SET n.signal = initial_signal + (feedback * $damping_factor)
            RETURN n.signal as new_signal, feedback as feedback_amount
        """, 
        source_id=source_element_id, 
        initial_signal=initial_signal,
        damping_factor=damping_factor)
        
        record = result.single()
        if record:
            feedback = record['feedback_amount']
            if abs(feedback) > 0.001:
                print(f"  🔄 Feedback damping applied to source: {feedback:+.4f} × {damping_factor} = {feedback * damping_factor:+.4f}")
    
    def _initialize_signal(self, session, ticker: str, magnitude: float):
        """Set initial signal on source node."""
        session.run("""
            MATCH (c:CompanyCanonical {ticker: $ticker})
            SET c.signal = $magnitude,
                c.signal_updated = datetime($timestamp)
        """,
        ticker=ticker,
        magnitude=magnitude,
        timestamp=self.timestamp)
        
        print(f"✓ Initialized signal on {ticker}: {magnitude:+.4f}")
    
    def _propagate_one_hop(self, 
                          session,
                          min_threshold: float,
                          shock_type: str,
                          exclude_element_id: str = None) -> Dict:
        """Propagate signals one hop from all active nodes."""
        
        # Build query with optional exclusion of source node
        where_clauses = [
            "source.signal IS NOT NULL",
            "source.signal <> 0",
            "abs(source.signal) >= $min_threshold"
        ]
        
        if exclude_element_id:
            where_clauses.append("elementId(target) <> $exclude_id")
        
        query = f"""
        MATCH (source)-[r]->(target)
        WHERE {' AND '.join(where_clauses)}
        RETURN 
            elementId(source) as source_id,
            labels(source)[0] as source_type,
            COALESCE(source.name, source.ticker) as source_name,
            source.signal as source_signal,
            elementId(target) as target_id,
            labels(target)[0] as target_type,
            COALESCE(target.name, target.ticker) as target_name,
            type(r) as rel_type,
            COALESCE(r.weight, 0.5) as edge_weight,
            COALESCE(r.confidence, 0.8) as edge_confidence,
            COALESCE(r.direction_value, 0) as direction_value,
            COALESCE(r.decay_rate, 0.0) as decay_rate
        """
        
        params = {'min_threshold': min_threshold}
        if exclude_element_id:
            params['exclude_id'] = exclude_element_id
        
        results = session.run(query, **params)
        
        target_signals = defaultdict(lambda: {
            'signal': 0.0,
            'source_count': 0,
            'paths': []
        })
        
        target_metadata = {}
        
        for record in results:
            source_signal = record['source_signal']
            edge_weight = record['edge_weight']
            edge_confidence = record['edge_confidence']
            direction_value = record['direction_value']
            rel_type = record['rel_type']
            
            propagation_multiplier = self._calculate_propagation_multiplier(
                rel_type=rel_type,
                edge_weight=edge_weight,
                edge_confidence=edge_confidence,
                direction_value=direction_value,
                shock_type=shock_type
            )
            
            propagated_signal = source_signal * propagation_multiplier
            
            decay_rate = record['decay_rate']
            if decay_rate > 0:
                decay_factor = math.exp(-decay_rate * 1.0)
                propagated_signal *= decay_factor
            
            target_id = record['target_id']
            target_signals[target_id]['signal'] += propagated_signal
            target_signals[target_id]['source_count'] += 1
            target_signals[target_id]['paths'].append({
                'from': record['source_name'],
                'relationship': rel_type,
                'contribution': propagated_signal
            })
            
            if target_id not in target_metadata:
                target_metadata[target_id] = {
                    'type': record['target_type'],
                    'name': record['target_name']
                }
        
        affected_nodes = {}
        
        for target_id, signal_data in target_signals.items():
            total_signal = signal_data['signal']
            
            session.run("""
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n.signal = COALESCE(n.signal, 0.0) + $signal,
                    n.signal_updated = datetime($timestamp)
            """,
            node_id=target_id,
            signal=total_signal,
            timestamp=self.timestamp)
            
            metadata = target_metadata[target_id]
            affected_nodes[target_id] = {
                'name': metadata['name'],
                'type': metadata['type'],
                'signal': total_signal,
                'source_count': signal_data['source_count'],
                'top_paths': sorted(signal_data['paths'], 
                                   key=lambda x: abs(x['contribution']), 
                                   reverse=True)[:3],
                'hop': 1
            }
            
            print(f"  {metadata['type']:15} | {metadata['name']:30} | {total_signal:+.4f}")
        
        return affected_nodes
    
    def _calculate_propagation_multiplier(self,
                                         rel_type: str,
                                         edge_weight: float,
                                         edge_confidence: float,
                                         direction_value: float,
                                         shock_type: str) -> float:
        """Calculate propagation multiplier based on relationship type."""
        
        base_multiplier = edge_weight * edge_confidence
        
        if rel_type == "SUPPLIES":
            if shock_type == "supply_disruption":
                return base_multiplier * 1.2
            else:
                return base_multiplier
        
        elif rel_type in ["PRODUCES", "REQUIRED_BY"]:
            if direction_value != 0:
                return base_multiplier * abs(direction_value)
            else:
                return base_multiplier * 0.5
        
        elif rel_type == "INFLUENCES":
            return base_multiplier
        
        elif rel_type == "BELONGS_TO":
            return base_multiplier * 0.7
        
        elif rel_type == "LOCATED_IN":
            if shock_type == "regulatory_change":
                return base_multiplier * 1.3
            else:
                return base_multiplier * 0.8
        
        else:
            return base_multiplier
    
    def _generate_summary(self, session, affected_nodes: Dict, hop_summary: List) -> Dict:
        """Generate summary statistics using actual current signals from database."""
        
        if not affected_nodes:
            return {
                'total_affected': 0,
                'by_type': {},
                'by_hop': [],
                'most_affected': []
            }
        
        # Get current signals from database for accuracy
        node_ids = list(affected_nodes.keys())
        current_signals = {}
        
        for node_id in node_ids:
            result = session.run("""
                MATCH (n)
                WHERE elementId(n) = $node_id
                RETURN COALESCE(n.signal, 0.0) as signal
            """, node_id=node_id)
            record = result.single()
            if record:
                current_signals[node_id] = record['signal']
        
        # Update affected_nodes with current signals
        for node_id, signal in current_signals.items():
            if node_id in affected_nodes:
                affected_nodes[node_id]['signal'] = signal
        
        # Count by node type
        by_type = defaultdict(int)
        for node in affected_nodes.values():
            by_type[node['type']] += 1
        
        most_affected = sorted(
            affected_nodes.values(),
            key=lambda x: abs(x['signal']),
            reverse=True
        )[:10]
        
        summary = {
            'total_affected': len(affected_nodes),
            'by_type': dict(by_type),
            'by_hop': hop_summary,
            'most_affected': [
                {
                    'name': n['name'],
                    'type': n['type'],
                    'signal': n['signal'],
                    'top_path': n['top_paths'][0] if n['top_paths'] else None
                }
                for n in most_affected
            ]
        }
        
        print(f"\n{'='*80}")
        print(f"PROPAGATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Affected Nodes: {summary['total_affected']}")
        print(f"\nBy Node Type:")
        for node_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {node_type:20} {count:4} nodes")
        
        print(f"\nTop 10 Most Affected:")
        for i, node in enumerate(most_affected, 1):
            print(f"  {i:2}. {node['name']:30} ({node['type']:15}) {node['signal']:+.4f}")
        
        return summary
    
    def reset_signals(self):
        """Clear all propagation signals."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.signal IS NOT NULL
                REMOVE n.signal, n.signal_updated
                RETURN count(n) as cleared
            """)
            
            count = result.single()['cleared']
            print(f"✓ Cleared signals from {count} nodes")


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    propagator = ModernPropagation(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )
    
    try:
        propagator.reset_signals()
        
        # Track specific entities through propagation
        entities_to_track = ["QS", "TSLA", "SAMSUNG_AUTO", "F"]
        
        print("\n" + "="*80)
        print("SCENARIO: Supply Chain Disruption with Entity Tracking")
        print("="*80)
        
        # Example 1: Realistic mode (default) - allows dampened feedback
        print("\n--- MODE: REALISTIC (Dampened Feedback) ---")
        propagator.reset_signals()
        
        result = propagator.propagate_shock(
            source_ticker="SAMSUNG_AUTO",
            shock_magnitude=-0.8,
            shock_type="supply_disruption",
            max_hops=3,
            track_entities=entities_to_track,
            feedback_mode="realistic"  # Allows 30% of feedback
        )
        
        propagator.print_entity_evolution_report(entities_to_track)
        
        # Example 2: Isolated mode - no feedback to source
        print("\n\n" + "="*80)
        print("\n--- MODE: ISOLATED (No Feedback) ---")
        propagator.reset_signals()
        
        result2 = propagator.propagate_shock(
            source_ticker="SAMSUNG_AUTO",
            shock_magnitude=-0.8,
            shock_type="supply_disruption",
            max_hops=3,
            track_entities=entities_to_track,
            feedback_mode="isolated"  # Source remains at initial shock
        )
        
        propagator.print_entity_evolution_report(entities_to_track)
        
        # Example 3: Full feedback - see the amplification effect
        print("\n\n" + "="*80)
        print("\n--- MODE: FULL (Unrestricted Feedback) ---")
        propagator.reset_signals()
        
        result3 = propagator.propagate_shock(
            source_ticker="SAMSUNG_AUTO",
            shock_magnitude=-0.8,
            shock_type="supply_disruption",
            max_hops=3,
            track_entities=entities_to_track,
            feedback_mode="full"  # Can amplify significantly
        )
        
        # Print detailed evolution report
        propagator.print_entity_evolution_report(entities_to_track)
        
        # Access programmatically
        print("\n\nProgrammatic Access:")
        for entity_id, evolution in result['summary'].get('entity_evolution', {}).items():
            print(f"\n{entity_id}:")
            print(f"  Total Change: {evolution['total_change']:+.4f}")
            print(f"  Final Signal: {evolution['final_signal']:+.4f}")
        
    finally:
        propagator.close()