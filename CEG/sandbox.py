from graph.signal_propagation import BoundedPropagation

# 1. Initialize
propagator = BoundedPropagation(uri="bolt://127.0.0.1:7687", user="neo4j", password="myhome2911!")

# 2. Propagate a shock
result = propagator.propagate_shock(
    events=[
        {'ticker': 'QS', 'magnitude': -0.3, 'type': 'supply_disruption'},
        {'ticker': 'TSLA', 'magnitude': -0.3, 'type': 'demand_surge'},
        {'ticker': 'A', 'magnitude': 1, 'type': 'supply_disruption'}
    ],
    max_hops=3,
    track_entities=["QS", "TSLA", "A", "ENPH"],
    feedback_mode="realistic",
)


# 4. Clear signals when done
propagator.reset_signals()