from graph.operators import add_ticker, propagate, reset_signals
from graph.neo4j_exporter import to_neo4j_enhanced
from graph.static_edges import StaticEdges

# # Generate edges for Boeing
# print("Generating edges...")
# static_edges = StaticEdges("BA")
# edges = static_edges.edges()

# print(f"Generated {len(edges)} edges")
# print("Exporting to Neo4j with enhanced properties...")

# # Export with enhanced properties
# success = to_neo4j_enhanced(
#     edges, 
#     "BA",
#     uri="bolt://127.0.0.1:7687",
#     user="neo4j",
#     password="myhome2911!"  # YOUR PASSWORD HERE
# )

# if success:
#     print("✅ Export complete!")
# else:
#     print("❌ Export failed")
    
    
    
    
# 1. Add companies to your graph
# add_ticker("TSLA")    # Boeing


# add_ticker("TSLA")  # Tesla
# add_ticker("AAPL")  # Apple

# 2. Simulate an event and propagate
results = propagate(
    source_ticker="QS",
    signal=0.4,           # Strong negative signal
    max_hops=3,            # Propagate 3 levels deep
    confidence_threshold=0.5  # Only use high-confidence edges
)

# 3. Check the results
print(f"Total affected: {results['summary']['total_affected']}")
print("\nMost impacted entities:")
for node in results['summary']['most_affected'][:5]:
    print(f"  {node['name']}: {node['signal']:.3f}")


results = propagate(
    source_ticker="BA",
    signal=-0.9,           # Strong negative signal
    max_hops=3,            # Propagate 3 levels deep
    confidence_threshold=0.5  # Only use high-confidence edges
)

# 3. Check the results
print(f"Total affected: {results['summary']['total_affected']}")
print("\nMost impacted entities:")
for node in results['summary']['most_affected'][:5]:
    print(f"  {node['name']}: {node['signal']:.3f}")
    
    
# 4. Reset when done
reset_signals()