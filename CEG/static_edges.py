import yfinance as yf
from suppliers import scrape_importyeti_suppliers
from commodities_engine.commodities import Commodities
from neo4j_ops import to_neo4j
import json

class StaticEdges:
    def __init__(self, ticker):
        self.ticker = ticker
        self.country = self.get_location()
        self.sector = self.get_sector()
        self.url = f"https://www.importyeti.com/company/tesla"
        self.suppliers = self.get_suppliers()
        self.produces = Commodities(ticker).produces
        self.requires = Commodities(ticker).requires
    
    def get_location(self):
        info = yf.Ticker(self.ticker).info
        country = info.get("country", "Unknown")
        return [country]

    def get_sector(self):
        info = yf.Ticker(self.ticker).info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        return [sector, industry]
    
    def get_suppliers(self):
        return scrape_importyeti_suppliers(self.url)
    
    def edges(self):
        """
        Returns edges in standardized JSON format:
        {
            "type": edge type (country, sector, supplier, produces, requires),
            "magnitude": impact magnitude (normalized value or boilerplate),
            "relevance": confidence/relevance score (0-1),
            "direction": flow direction (company->target or target->company),
            "data": edge-specific additional data
        }
        
        Bidirectional relationships create two edges with different magnitudes/relevance.
        """
        edges_list = []
        
        # Get company market cap for impact calculations
        info = yf.Ticker(self.ticker).info
        market_cap = info.get("marketCap", 0)
        
        # Country edges (BIDIRECTIONAL)
        for country in self.country:
            # Company operates in country
            edges_list.append({
                "type": "country",
                "magnitude": 1.0,
                "relevance": 1.0,
                "direction": "country->company",
                "data": {
                    "country_name": country,
                    "relationship": "headquarters",
                    "impact_type": "regulatory_environment"
                }
            })
            
            # Company impacts country
            if market_cap > 1e12:
                country_impact = 0.85
            elif market_cap > 1e11:
                country_impact = 0.70
            elif market_cap > 1e10:
                country_impact = 0.50
            else:
                country_impact = 0.30
            
            edges_list.append({
                "type": "country",
                "magnitude": country_impact,
                "relevance": 0.80,
                "direction": "company->country",
                "data": {
                    "country_name": country,
                    "relationship": "economic_contributor",
                    "impact_type": "employment_tax_innovation",
                    "market_cap": market_cap
                }
            })
        
        # Sector edges (BIDIRECTIONAL)
        sector_info = self.sector
        if len(sector_info) >= 1 and sector_info[0] != "Unknown":
            edges_list.append({
                "type": "sector",
                "magnitude": 0.90,
                "relevance": 1.0,
                "direction": "sector->company",
                "data": {
                    "sector_name": sector_info[0],
                    "classification_level": "sector",
                    "impact_type": "market_trends_regulations"
                }
            })
            
            if market_cap > 5e11:
                sector_impact = 0.95
            elif market_cap > 1e11:
                sector_impact = 0.75
            elif market_cap > 1e10:
                sector_impact = 0.55
            else:
                sector_impact = 0.35
            
            edges_list.append({
                "type": "sector",
                "magnitude": sector_impact,
                "relevance": 0.85,
                "direction": "company->sector",
                "data": {
                    "sector_name": sector_info[0],
                    "classification_level": "sector",
                    "impact_type": "innovation_disruption",
                    "market_cap": market_cap
                }
            })
        
        # Industry edges (BIDIRECTIONAL)
        if len(sector_info) >= 2 and sector_info[1] != "Unknown":
            edges_list.append({
                "type": "industry",
                "magnitude": 0.95,
                "relevance": 1.0,
                "direction": "industry->company",
                "data": {
                    "industry_name": sector_info[1],
                    "classification_level": "industry",
                    "impact_type": "competitive_dynamics"
                }
            })
            
            if market_cap > 5e11:
                industry_impact = 0.90
            elif market_cap > 1e11:
                industry_impact = 0.70
            elif market_cap > 1e10:
                industry_impact = 0.50
            else:
                industry_impact = 0.30
            
            edges_list.append({
                "type": "industry",
                "magnitude": industry_impact,
                "relevance": 0.85,
                "direction": "company->industry",
                "data": {
                    "industry_name": sector_info[1],
                    "classification_level": "industry",
                    "impact_type": "market_share_innovation",
                    "market_cap": market_cap
                }
            })
        
        # Supplier edges (BIDIRECTIONAL)
        if self.suppliers:
            max_shipments = max([int(s.get('total_shipments', '0').replace(',', '')) 
                                for s in self.suppliers if s.get('total_shipments')] + [1])
            
            for supplier in self.suppliers:
                try:
                    shipments = int(supplier.get('total_shipments', '0').replace(',', ''))
                except:
                    shipments = 0
                
                supply_magnitude = shipments / max_shipments if max_shipments > 0 else 0.5
                
                edges_list.append({
                    "type": "supplier",
                    "magnitude": round(supply_magnitude, 4),
                    "relevance": 0.85,
                    "direction": "supplier->company",
                    "data": {
                        "supplier_name": supplier.get('supplier_name', ''),
                        "supplier_url": supplier.get('supplier_url', ''),
                        "location": supplier.get('location', ''),
                        "total_shipments": supplier.get('total_shipments', ''),
                        "product_description": supplier.get('product_description', ''),
                        "hs_codes": supplier.get('hs_codes', ''),
                        "impact_type": "supply_provision"
                    }
                })
                
                edges_list.append({
                    "type": "supplier",
                    "magnitude": round(supply_magnitude * 0.8, 4),
                    "relevance": 0.75,
                    "direction": "company->supplier",
                    "data": {
                        "supplier_name": supplier.get('supplier_name', ''),
                        "supplier_url": supplier.get('supplier_url', ''),
                        "location": supplier.get('location', ''),
                        "total_shipments": supplier.get('total_shipments', ''),
                        "product_description": supplier.get('product_description', ''),
                        "hs_codes": supplier.get('hs_codes', ''),
                        "impact_type": "demand_generation"
                    }
                })
        
        # Produces edges
        for produce in self.produces:
            edges_list.append({
                "type": "produces",
                "magnitude": round(produce.get('production_value', 0), 6),
                "relevance": round(produce.get('confidence', 0.5), 4),
                "direction": "company->commodity",
                "data": {
                    "naics_code": produce.get('naics_code', ''),
                    "description": produce.get('description', ''),
                    "commodity_type": "output"
                }
            })
        
        # Requires edges
        for require in self.requires:
            edges_list.append({
                "type": "requires",
                "magnitude": round(require.get('requirement_value', 0), 6),
                "relevance": round(require.get('confidence', 0.5), 4),
                "direction": "commodity->company",
                "data": {
                    "naics_code": require.get('naics_code', ''),
                    "description": require.get('description', ''),
                    "layer": require.get('layer', 'unknown'),
                    "commodity_type": "input"
                }
            })
        
        return edges_list
    
    def to_neo4j(self, uri="neo4j://127.0.0.1:7687", user="neo4j", password="myhome2911!"):
        to_neo4j(self.edges(), self.ticker, uri, user, password)



# Example usage
if __name__ == "__main__":
    static_edges = StaticEdges("TSLA")
    
    # Option 1: Get edges as JSON
    edges = static_edges.edges()
    
    from collections import Counter
    edge_types = Counter([e['type'] for e in edges])
    
    print(f"\n📊 Total Edges: {len(edges)}")
    print(f"Edge Types: {dict(edge_types)}\n")
    
    # Save to JSON
    with open('tesla_edges.json', 'w') as f:
        json.dump(edges, f, indent=2)
    print("✅ Full edges saved to 'tesla_edges.json'")
    
    # Option 2: Export to Neo4j
    print("\n" + "=" * 80)
    print("Exporting to Neo4j...")
    print("=" * 80)
    
    success = static_edges.to_neo4j(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"
    )
    