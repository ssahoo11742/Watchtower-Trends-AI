from graph.signal_propagation import ModernPropagation

# 1. Initialize
propagator = ModernPropagation(uri="bolt://127.0.0.1:7687", user="neo4j", password="myhome2911!")

# 2. Propagate a shock
result = propagator.propagate_shock(
    source_ticker="QS",
    shock_magnitude=-0.7,  # -1.0 to 1.0
    shock_type="supply_disruption",
    max_hops=3,
    track_entities=["ENPH"],
    feedback_mode="realistic"
)



# 4. Clear signals when done
propagator.reset_signals()