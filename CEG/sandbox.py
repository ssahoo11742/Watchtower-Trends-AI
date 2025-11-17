from graph.signal_propagation import BoundedPropagation
# from graph.static_edges import StaticEdges

# 


# 1. Initialize
propagator = BoundedPropagation(uri="bolt://127.0.0.1:7687", user="neo4j", password="myhome2911!")
# Value rotation: old economy up, new economy down
# QS has breakthrough, threatens Tesla's battery plans
result = propagator.propagate_shock(
    events=[
        {'ticker': 'QS', 'magnitude': 0.9, 'type': 'breakthrough_technology'},
        {'ticker': 'TSLA', 'magnitude': -0.3, 'type': 'competitive_threat'},
    ],
    max_hops=4,
    track_entities=["QS", "TSLA", "ENPH", "A", "BA", "JPM", "PG"],
    feedback_mode="realistic",
)
propagator.print_explanation_report()

# edges = StaticEdges(
#     "TSLA", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/tesla"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )

# edges = StaticEdges(
#     "A", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/Agilent-Technologies"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )


# edges = StaticEdges(
#     "ENPH", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/enphase-energy"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )


# edges = StaticEdges(
#     "JPM", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/jpmorgan-chase-bank-n-a"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )

# edges = StaticEdges(
#     "BA", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/boeing-commercial-airp"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )


# edges = StaticEdges(
#     "PG", 
#     compute_correlations=True,
#     supplier_url="https://www.importyeti.com/company/procter-gamble"
# )

# edges.to_neo4j(
#     canonical_companies_csv="./data/companies_filtered.csv",
#     auto_canonicalize=True,
#     min_occurrences=2,
#     password="myhome2911!",
# )




# 4. Clear signals when done
propagator.reset_signals()