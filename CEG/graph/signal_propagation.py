"""
Bounded Signal Propagation Engine for Neo4j Graph (Bidirectional)
=================================================================

Signals represent normalized impact scores in range [-1, 1]:
- -1.0 = Maximum negative impact (stock expected to do very badly)
- 0.0 = No impact (neutral)
- +1.0 = Maximum positive impact (stock expected to do very well)

Signals propagate in BOTH directions:
- Forward: Through outgoing relationships (e.g., SUPPLIES, PRODUCES)
- Backward: Through incoming REQUIRED_BY (splits signal among suppliers)
"""

from neo4j import GraphDatabase
from datetime import datetime
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BoundedPropagation:
    """
    Propagates bounded sentiment signals through knowledge graph.
    All signals stay within [-1, 1] range.
    Supports bidirectional propagation.
    """
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.timestamp = datetime.now().isoformat()
        self.entity_tracker = defaultdict(list)
    
    def close(self):
        self.driver.close()
    
    @staticmethod
    def combine_signals(signals: List[float]) -> float:
        """
        Combine multiple signals using probabilistic fusion.
        
        This ensures:
        1. Result stays in [-1, 1]
        2. Multiple weak signals can't overpower a strong signal
        3. Conflicting signals partially cancel out
        
        Uses modified opinion pooling formula.
        """
        if not signals:
            return 0.0
        
        if len(signals) == 1:
            return max(-1.0, min(1.0, signals[0]))
        
        # Separate positive and negative signals
        positive = [s for s in signals if s > 0]
        negative = [s for s in signals if s < 0]
        
        # Combine same-direction signals with diminishing returns
        def combine_same_direction(sigs):
            if not sigs:
                return 0.0
            # Sort by magnitude
            sigs = sorted(sigs, key=abs, reverse=True)
            result = sigs[0]
            for sig in sigs[1:]:
                # Each additional signal has diminishing impact
                result = result + sig * (1 - abs(result))
            return result
        
        pos_combined = combine_same_direction(positive)
        neg_combined = combine_same_direction(negative)
        
        # Combine opposing signals (they partially cancel)
        final = pos_combined + neg_combined
        
        # Ensure bounded
        return max(-1.0, min(1.0, final))
    
    def propagate_shock(self, 
                       source_ticker: str = None,
                       shock_magnitude: float = None,
                       shock_type: str = "supply_disruption",
                       max_hops: int = 3,
                       min_propagation_threshold: float = 0.01,
                       track_entities: List[str] = None,
                       feedback_mode: str = "realistic",
                       events: List[Dict] = None) -> Dict:
        """
        Propagate bounded shock(s) through the network (bidirectional).
        
        Args:
            source_ticker: Single source (deprecated, use events instead)
            shock_magnitude: Single magnitude (deprecated, use events instead)
            shock_type: Type of shock (used for all events if not specified per-event)
            max_hops: Maximum propagation distance
            min_propagation_threshold: Minimum signal strength to continue
            track_entities: List of tickers/names to track at each hop
            feedback_mode: "realistic" (dampened), "isolated" (none), "full" (unlimited)
            events: List of events, e.g.:
                    [
                        {'ticker': 'QS', 'magnitude': -0.7, 'type': 'supply_disruption'},
                        {'ticker': 'TSLA', 'magnitude': 0.1, 'type': 'demand_surge'},
                        {'ticker': 'A', 'magnitude': -0.2}
                    ]
        
        Returns:
            Dict with propagation results
        """
        
        # Handle both old single-event and new multi-event API
        if events is None:
            if source_ticker is None or shock_magnitude is None:
                raise ValueError("Must provide either (source_ticker, shock_magnitude) or events list")
            events = [{'ticker': source_ticker, 'magnitude': shock_magnitude, 'type': shock_type}]
        
        # Validate all event magnitudes
        for event in events:
            mag = event['magnitude']
            if not -1.0 <= mag <= 1.0:
                raise ValueError(f"magnitude must be in [-1, 1], got {mag} for {event['ticker']}")
        
        print(f"\n{'='*80}")
        print(f"🚨 BOUNDED SHOCK PROPAGATION (BIDIRECTIONAL): {len(events)} Event(s)")
        print(f"{'='*80}")
        for event in events:
            ticker = event['ticker']
            magnitude = event['magnitude']
            event_type = event.get('type', shock_type)
            print(f"  • {ticker:10} {magnitude:+.4f}  ({event_type})")
        print(f"\nMax Hops: {max_hops}")
        print(f"Feedback Mode: {feedback_mode}")
        if track_entities:
            print(f"Tracking: {', '.join(track_entities)}")
        
        with self.driver.session() as session:
            self.entity_tracker.clear()
            
            # Get source node elementIds for all events
            source_element_ids = {}
            initial_signals = {}
            
            for event in events:
                ticker = event['ticker']
                magnitude = event['magnitude']
                
                result = session.run("""
                    MATCH (c:CompanyCanonical {ticker: $ticker})
                    RETURN elementId(c) as element_id
                """, ticker=ticker)
                record = result.single()
                if record:
                    source_element_ids[ticker] = record['element_id']
                    initial_signals[ticker] = magnitude
                else:
                    print(f"⚠️  Warning: {ticker} not found in graph")
            
            # Initialize all source nodes with their signals
            for event in events:
                ticker = event['ticker']
                magnitude = event['magnitude']
                if ticker in source_element_ids:
                    self._initialize_signal(session, ticker, magnitude)
            
            # Track initial state
            if track_entities:
                self._snapshot_entities(session, track_entities, hop=0, phase="initial")
            
            all_affected = {}
            hop_summary = []
            
            # Propagate through hops
            for hop in range(1, max_hops + 1):
                print(f"\n{'─'*80}")
                print(f"HOP {hop}/{max_hops}")
                print(f"{'─'*80}")
                
                if track_entities:
                    self._snapshot_entities(session, track_entities, hop=hop, phase="before")
                
                affected_nodes = self._propagate_one_hop_bidirectional(
                    session,
                    min_threshold=min_propagation_threshold,
                    shock_type=shock_type,
                    exclude_element_ids=set(source_element_ids.values()) if feedback_mode == "isolated" else None,
                    hop_number=hop
                )
                
                # Apply feedback damping for realistic mode (for all source nodes)
                if feedback_mode == "realistic":
                    for ticker, element_id in source_element_ids.items():
                        self._apply_feedback_damping(
                            session, 
                            element_id, 
                            initial_signals[ticker],
                            damping_factor=0.1
                        )
                
                if track_entities:
                    self._snapshot_entities(session, track_entities, hop=hop, phase="after")
                    self._print_entity_changes(session, track_entities, hop)
                
                if not affected_nodes:
                    print(f"✓ No more nodes affected. Stopping at hop {hop}.")
                    break
                
                # Update aggregate results
                for node_id, info in affected_nodes.items():
                    if node_id not in all_affected:
                        all_affected[node_id] = info
                    else:
                        # Already tracked
                        pass
                
                # After feedback damping, update source signals in affected_nodes
                if feedback_mode == "realistic":
                    for ticker, element_id in source_element_ids.items():
                        actual_signal = session.run("""
                            MATCH (n)
                            WHERE elementId(n) = $source_id
                            RETURN n.signal as signal
                        """, source_id=element_id).single()
                        
                        if actual_signal and element_id in all_affected:
                            all_affected[element_id]['signal'] = actual_signal['signal']
                
                hop_summary.append({
                    'hop': hop,
                    'nodes_affected': len(affected_nodes)
                })
                
                print(f"  → {len(affected_nodes)} nodes affected")
            
            # Generate summary
            summary = self._generate_summary(session, all_affected, hop_summary)
            
            if track_entities:
                summary['entity_evolution'] = self._get_entity_evolution(track_entities)
            
            return {
                'events': events,
                'shock_type': shock_type,
                'affected_nodes': all_affected,
                'summary': summary,
                'entity_tracking': dict(self.entity_tracker) if track_entities else {}
            }
    
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
    
    def _propagate_one_hop_bidirectional(self, 
                          session,
                          min_threshold: float,
                          shock_type: str,
                          exclude_element_ids: set = None,
                          hop_number: int = 1) -> Dict:
        """
        Propagate bounded signals one hop in BOTH directions.
        
        Forward: (source)-[r]->(target) - normal propagation
        Backward: (source)<-[REQUIRED_BY]-(supplier) - split signal among suppliers
        """
        
        # Attenuation factor: signals weaken over distance
        distance_decay = 0.7 ** (hop_number - 1)  # 70% per hop
        
        where_clauses = [
            "source.signal IS NOT NULL",
            "source.signal <> 0",
            "abs(source.signal) >= $min_threshold"
        ]
        
        if exclude_element_ids:
            where_clauses.append("NOT elementId(target) IN $exclude_ids")
        
        # ===== FORWARD PROPAGATION =====
        forward_query = f"""
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
            COALESCE(target.ticker, target.name) as target_identifier,
            type(r) as rel_type,
            COALESCE(r.weight, 0.5) as edge_weight,
            COALESCE(r.confidence, 0.8) as edge_confidence,
            COALESCE(r.direction_value, 0) as direction_value,
            'forward' as propagation_direction
        """
        
        # ===== BACKWARD PROPAGATION =====
        # 1. Through RESOLVES_TO: CompanyCanonical <- surface company (propagate TO surface)
        # 2. Through REQUIRED_BY: Company -> suppliers
        # 3. Through SUPPLIES: Company <- suppliers (when company is disrupted, suppliers affected)
        
        backward_query = f"""
        // Part 1: RESOLVES_TO relationships (surface -> canonical, propagate TO surface)
        MATCH (surface)-[r:RESOLVES_TO]->(source)
        WHERE {' AND '.join(where_clauses.copy())}
        RETURN 
            elementId(source) as source_id,
            labels(source)[0] as source_type,
            COALESCE(source.name, source.ticker) as source_name,
            source.signal as source_signal,
            elementId(surface) as target_id,
            labels(surface)[0] as target_type,
            COALESCE(surface.name, surface.ticker) as target_name,
            COALESCE(surface.ticker, surface.name) as target_identifier,
            'RESOLVES_TO' as rel_type,
            COALESCE(r.weight, 0.5) as edge_weight,
            COALESCE(r.confidence, 0.8) as edge_confidence,
            COALESCE(r.direction_value, 0) as direction_value,
            'backward' as propagation_direction,
            1 as split_count
        
        UNION ALL
        
        // Part 2: SUPPLIES relationships (supplier -> company, propagate TO supplier)
        // When a company is disrupted, its suppliers are affected
        MATCH (supplier)-[r:SUPPLIES]->(source)
        WHERE {' AND '.join(where_clauses.copy())}
        WITH source, COUNT(DISTINCT supplier) as total_suppliers
        MATCH (supplier)-[r:SUPPLIES]->(source)
        WHERE {' AND '.join(where_clauses.copy())}
        RETURN 
            elementId(source) as source_id,
            labels(source)[0] as source_type,
            COALESCE(source.name, source.ticker) as source_name,
            source.signal as source_signal,
            elementId(supplier) as target_id,
            labels(supplier)[0] as target_type,
            COALESCE(supplier.name, supplier.ticker) as target_name,
            COALESCE(supplier.ticker, supplier.name) as target_identifier,
            'SUPPLIES' as rel_type,
            COALESCE(r.weight, 0.5) as edge_weight,
            COALESCE(r.confidence, 0.8) as edge_confidence,
            COALESCE(r.direction_value, 0) as direction_value,
            'backward' as propagation_direction,
            total_suppliers as split_count
        
        UNION ALL
        
        // Part 3: REQUIRED_BY relationships (company -> suppliers)
        MATCH (supplier)-[r:REQUIRED_BY]->(source)
        WHERE {' AND '.join(where_clauses.copy())}
        WITH source, COUNT(DISTINCT supplier) as total_suppliers
        MATCH (supplier)-[r:REQUIRED_BY]->(source)
        WHERE {' AND '.join(where_clauses.copy())}
        RETURN 
            elementId(source) as source_id,
            labels(source)[0] as source_type,
            COALESCE(source.name, source.ticker) as source_name,
            source.signal as source_signal,
            elementId(supplier) as target_id,
            labels(supplier)[0] as target_type,
            COALESCE(supplier.name, supplier.ticker) as target_name,
            COALESCE(supplier.ticker, supplier.name) as target_identifier,
            'REQUIRED_BY' as rel_type,
            COALESCE(r.weight, 0.5) as edge_weight,
            COALESCE(r.confidence, 0.8) as edge_confidence,
            COALESCE(r.direction_value, 0) as direction_value,
            'backward' as propagation_direction,
            total_suppliers as split_count
        """
        
        params = {'min_threshold': min_threshold}
        if exclude_element_ids:
            params['exclude_ids'] = list(exclude_element_ids)
        
        # Execute both queries
        forward_results = session.run(forward_query, **params)
        backward_results = session.run(backward_query, **params)
        
        # Group incoming signals by target
        target_signals = defaultdict(lambda: {
            'signals': [],
            'source_count': 0,
            'paths': []
        })
        
        target_metadata = {}
        
        # Process FORWARD propagation
        print("\n  [FORWARD PROPAGATION]")
        for record in forward_results:
            self._process_propagation_record(
                record, target_signals, target_metadata, 
                distance_decay, shock_type, hop_number,
                split_factor=1.0  # No splitting for forward
            )
        
        # Process BACKWARD propagation
        print("\n  [BACKWARD PROPAGATION - through RESOLVES_TO and SUPPLIES to suppliers]")
        for record in backward_results:
            split_count = record['split_count']
            split_factor = 1.0 / split_count if split_count > 0 else 1.0
            
            rel_type = record['rel_type']
            if rel_type == 'RESOLVES_TO':
                print(f"    RESOLVES_TO: {record['source_name']} -> {record['target_name']}")
            elif rel_type == 'SUPPLIES' and split_count > 1:
                print(f"    SUPPLIES (backward): Splitting signal among {split_count} suppliers (factor: {split_factor:.3f})")
            elif rel_type == 'REQUIRED_BY' and split_count > 1:
                print(f"    REQUIRED_BY: Splitting signal among {split_count} suppliers (factor: {split_factor:.3f})")
            
            self._process_propagation_record(
                record, target_signals, target_metadata, 
                distance_decay, shock_type, hop_number,
                split_factor=split_factor
            )
        
        # Combine signals at each target using bounded fusion
        affected_nodes = {}
        
        for target_id, signal_data in target_signals.items():
            # Combine multiple incoming signals
            combined_signal = self.combine_signals(signal_data['signals'])
            
            # Get current signal (for accumulation across hops)
            current = session.run("""
                MATCH (n)
                WHERE elementId(n) = $node_id
                RETURN COALESCE(n.signal, 0.0) as current_signal
            """, node_id=target_id).single()
            
            current_signal = current['current_signal'] if current else 0.0
            
            # Combine with existing signal
            final_signal = self.combine_signals([current_signal, combined_signal])
            
            # Update node
            session.run("""
                MATCH (n)
                WHERE elementId(n) = $node_id
                SET n.signal = $signal,
                    n.signal_updated = datetime($timestamp)
            """,
            node_id=target_id,
            signal=final_signal,
            timestamp=self.timestamp)
            
            metadata = target_metadata[target_id]
            affected_nodes[target_id] = {
                'name': metadata['name'],
                'type': metadata['type'],
                'signal': final_signal,
                'source_count': signal_data['source_count'],
                'top_paths': sorted(signal_data['paths'], 
                                   key=lambda x: abs(x['contribution']), 
                                   reverse=True)[:3],
                'hop': hop_number
            }
            
            print(f"  {metadata['type']:15} | {metadata['name']:30} | {final_signal:+.4f}")
        
        return affected_nodes
    
    def _process_propagation_record(self, record, target_signals, target_metadata, 
                                   distance_decay, shock_type, hop_number, split_factor=1.0):
        """Process a single propagation record (forward or backward)."""
        source_signal = record['source_signal']
        edge_weight = record['edge_weight']
        edge_confidence = record['edge_confidence']
        direction_value = record['direction_value']
        rel_type = record['rel_type']
        
        # Calculate propagation strength
        propagation_strength = self._calculate_propagation_multiplier(
            rel_type=rel_type,
            edge_weight=edge_weight,
            edge_confidence=edge_confidence,
            direction_value=direction_value,
            shock_type=shock_type
        )
        
        # Apply distance decay and split factor
        propagation_strength *= distance_decay * split_factor
        
        # Calculate propagated signal (stays bounded)
        propagated_signal = source_signal * propagation_strength
        
        # Ensure bounded
        propagated_signal = max(-1.0, min(1.0, propagated_signal))
        
        target_id = record['target_id']
        target_signals[target_id]['signals'].append(propagated_signal)
        target_signals[target_id]['source_count'] += 1
        target_signals[target_id]['paths'].append({
            'from': record['source_name'],
            'relationship': rel_type,
            'contribution': propagated_signal,
            'direction': record.get('propagation_direction', 'forward')
        })
        
        if target_id not in target_metadata:
            target_metadata[target_id] = {
                'type': record['target_type'],
                'name': record['target_name'],
                'identifier': record['target_identifier']
            }
    
    def _calculate_propagation_multiplier(self,
                                         rel_type: str,
                                         edge_weight: float,
                                         edge_confidence: float,
                                         direction_value: float,
                                         shock_type: str) -> float:
        """
        Calculate propagation strength (how much signal passes through edge).
        Returns value in [0, 1] range typically.
        """
        
        base_strength = edge_weight * edge_confidence
        
        if rel_type == "SUPPLIES":
            if shock_type == "supply_disruption":
                return base_strength * 0.05  # Strong but not amplifying
            else:
                return base_strength * 0.04
        
        elif rel_type in ["PRODUCES", "REQUIRED_BY", "RESOLVES_TO"]:
            if direction_value != 0:
                return base_strength * 0.7
            else:
                # Strong propagation through RESOLVES_TO (canonical -> entity)
                if rel_type == "RESOLVES_TO":
                    return base_strength * 0.9  # High fidelity for canonical resolution
                else:
                    return base_strength * 0.5  # Default for REQUIRED_BY backward propagation
        
        elif rel_type == "INFLUENCES":
            return base_strength * 0.6
        
        elif rel_type == "BELONGS_TO":
            return base_strength * 0.5
        
        elif rel_type == "LOCATED_IN":
            if shock_type == "regulatory_change":
                return base_strength * 0.7
            else:
                return base_strength * 0.5
        
        else:
            return base_strength * 0.5
    
    def _apply_feedback_damping(self, session, source_element_id: str, 
                               initial_signal: float, damping_factor: float = 0.1):
        """Apply conservative damping to feedback."""
        result = session.run("""
            MATCH (n)
            WHERE elementId(n) = $source_id
            WITH n, n.signal as current_signal, $initial_signal as initial_signal
            WITH n, current_signal, initial_signal,
                 current_signal - initial_signal as feedback
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
                print(f"  🔄 Feedback damped: {feedback:+.4f} × {damping_factor} = {feedback * damping_factor:+.4f}")
    
    def _snapshot_entities(self, session, entity_ids: List[str], hop: int, phase: str):
        """Capture current signal state for tracked entities."""
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
            node_id = record['node_id']
            
            # Get contributing paths if this is an "after" snapshot
            contributing_paths = []
            if phase == "after" and hop > 0:
                contributing_paths = self._get_contributing_paths(session, node_id, hop)
            
            self.entity_tracker[identifier].append({
                'hop': hop,
                'phase': phase,
                'signal': record['signal'],
                'name': record['name'],
                'type': record['type'],
                'node_id': record['node_id'],
                'contributing_paths': contributing_paths
            })
    
    def _get_contributing_paths(self, session, target_node_id: str, current_hop: int) -> List[Dict]:
        """Get the paths that contributed to this node's signal in the current hop."""
        
        # Find all nodes that propagated to this target
        query = """
        MATCH (source)-[r]->(target)
        WHERE elementId(target) = $target_id
        AND source.signal IS NOT NULL
        AND source.signal <> 0
        RETURN 
            COALESCE(source.ticker, source.name) as source_name,
            labels(source)[0] as source_type,
            source.signal as source_signal,
            type(r) as relationship,
            COALESCE(r.weight, 0.5) as weight,
            COALESCE(r.confidence, 0.8) as confidence
        ORDER BY abs(source.signal) DESC
        LIMIT 5
        
        UNION
        
        // Backward paths
        MATCH (source)<-[r]-(target)
        WHERE elementId(target) = $target_id
        AND source.signal IS NOT NULL
        AND source.signal <> 0
        AND type(r) IN ['REQUIRED_BY', 'SUPPLIES', 'RESOLVES_TO']
        RETURN 
            COALESCE(source.ticker, source.name) as source_name,
            labels(source)[0] as source_type,
            source.signal as source_signal,
            type(r) + ' (backward)' as relationship,
            COALESCE(r.weight, 0.5) as weight,
            COALESCE(r.confidence, 0.8) as confidence
        ORDER BY abs(source.signal) DESC
        LIMIT 5
        """
        
        results = session.run(query, target_id=target_node_id)
        
        paths = []
        for record in results:
            paths.append({
                'source': record['source_name'],
                'source_type': record['source_type'],
                'source_signal': record['source_signal'],
                'relationship': record['relationship'],
                'weight': record['weight'],
                'confidence': record['confidence']
            })
        
        return paths
    
    def _print_entity_changes(self, session, entity_ids: List[str], hop: int):
        """Print changes for tracked entities at this hop."""
        print(f"\n  📊 Tracked Entity Changes (Hop {hop}):")
        
        for entity_id in entity_ids:
            if entity_id not in self.entity_tracker:
                continue
            
            snapshots = self.entity_tracker[entity_id]
            before = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'before'), None)
            after = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'after'), None)
            
            if before and after:
                change = after['signal'] - before['signal']
                if abs(change) > 0.0001:
                    print(f"     {entity_id:15} | {before['signal']:+.6f} → {after['signal']:+.6f} (Δ {change:+.6f})")
                    
                    # Print explanation
                    if after.get('contributing_paths'):
                        print(f"       └─ Reasons:")
                        for path in after['contributing_paths'][:3]:  # Top 3 contributors
                            direction = "↓" if path['source_signal'] < 0 else "↑"
                            print(f"          • {path['source']} ({path['source_type']}) "
                                  f"{direction} {path['source_signal']:+.4f} via {path['relationship']}")
                else:
                    print(f"     {entity_id:15} | {before['signal']:+.6f} (no change)")
    
    def _get_entity_evolution(self, entity_ids: List[str]) -> Dict:
        """Get complete evolution summary for tracked entities."""
        evolution = {}
        
        for entity_id in entity_ids:
            if entity_id not in self.entity_tracker:
                evolution[entity_id] = {'error': 'Entity not found'}
                continue
            
            snapshots = self.entity_tracker[entity_id]
            initial = next((s for s in snapshots if s['phase'] == 'initial'), None)
            final = snapshots[-1] if snapshots else None
            
            hop_changes = []
            for hop_num in set(s['hop'] for s in snapshots if s['hop'] > 0):
                before = next((s for s in snapshots if s['hop'] == hop_num and s['phase'] == 'before'), None)
                after = next((s for s in snapshots if s['hop'] == hop_num and s['phase'] == 'after'), None)
                
                if before and after:
                    hop_changes.append({
                        'hop': hop_num,
                        'before': before['signal'],
                        'after': after['signal'],
                        'change': after['signal'] - before['signal'],
                        'contributing_paths': after.get('contributing_paths', [])
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
            
            initial_val = snapshots[0]['signal']
            final_val = snapshots[-1]['signal']
            total_change = final_val - initial_val
            
            print(f"\n  Summary:")
            print(f"    Initial:       {initial_val:+.4f}")
            print(f"    Final:         {final_val:+.4f}")
            print(f"    Total Change:  {total_change:+.4f}")
    
    def print_explanation_report(self, entity_ids: List[str] = None):
        """Print detailed explanation of why each tracked entity changed."""
        if not self.entity_tracker:
            print("No entities tracked in last propagation.")
            return
        
        entities_to_report = entity_ids if entity_ids else list(self.entity_tracker.keys())
        
        print(f"\n{'='*80}")
        print(f"SIGNAL CHANGE EXPLANATION REPORT")
        print(f"{'='*80}")
        
        for entity_id in entities_to_report:
            if entity_id not in self.entity_tracker:
                print(f"\n❌ {entity_id}: Not found")
                continue
            
            snapshots = self.entity_tracker[entity_id]
            entity_name = snapshots[0]['name']
            entity_type = snapshots[0]['type']
            
            print(f"\n{'─'*80}")
            print(f"📊 {entity_name} ({entity_type})")
            print(f"{'─'*80}")
            
            # Get initial and final states
            initial = next((s for s in snapshots if s['hop'] == 0), None)
            final = snapshots[-1] if snapshots else None
            
            if initial and final:
                total_change = final['signal'] - initial['signal']
                print(f"Initial Signal: {initial['signal']:+.6f}")
                print(f"Final Signal:   {final['signal']:+.6f}")
                print(f"Total Change:   {total_change:+.6f}")
            
            # Show hop-by-hop explanations
            hops = sorted(set(s['hop'] for s in snapshots if s['hop'] > 0))
            
            for hop in hops:
                before = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'before'), None)
                after = next((s for s in snapshots if s['hop'] == hop and s['phase'] == 'after'), None)
                
                if before and after:
                    change = after['signal'] - before['signal']
                    
                    if abs(change) > 0.0001:
                        print(f"\n  Hop {hop}: {before['signal']:+.6f} → {after['signal']:+.6f} (Δ {change:+.6f})")
                        
                        if after.get('contributing_paths'):
                            print(f"  Contributing factors:")
                            for i, path in enumerate(after['contributing_paths'], 1):
                                impact_direction = "negative" if path['source_signal'] < 0 else "positive"
                                print(f"    {i}. {path['source']} had {impact_direction} signal "
                                      f"({path['source_signal']:+.4f})")
                                print(f"       via {path['relationship']} relationship "
                                      f"(weight: {path['weight']:.2f}, confidence: {path['confidence']:.2f})")
                    else:
                        print(f"\n  Hop {hop}: No significant change")
    
    def _generate_summary(self, session, affected_nodes: Dict, hop_summary: List) -> Dict:
        """Generate summary statistics using actual current signals."""
        
        if not affected_nodes:
            return {
                'total_affected': 0,
                'by_type': {},
                'by_hop': [],
                'most_affected': []
            }
        
        # Get current signals from database
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
        
        print(f"\nTop 10 Most Affected (by absolute impact):")
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