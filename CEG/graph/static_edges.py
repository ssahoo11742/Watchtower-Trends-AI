import yfinance as yf
from suppliers.suppliers import scrape_importyeti_suppliers
from commodities_engine.commodities import Commodities
from .neo4j_exporter import to_neo4j_enhanced
from .correlation_computer import compute_correlation_strength
from .data_driven_metrics import DataDrivenMetrics  # NEW!
import json
import math
from collections import defaultdict
from .supplier_weight import SupplierWeights
class StaticEdges:
    def __init__(self, ticker, compute_correlations=True, supplier_url=None):
        """
        Initialize edge constructor with DATA-DRIVEN metrics.
        
        Args:
            ticker: Company ticker
            compute_correlations: If True, compute price correlations
            supplier_url: ImportYeti URL (if None, will try to construct from company name)
        """
        self.ticker = ticker
        self.compute_correlations = compute_correlations
        
        # Initialize data-driven metrics calculator
        self.metrics = DataDrivenMetrics(ticker, companies_csv="./data/companies.csv")
        
        self.country = self.get_location()
        self.sector = self.get_sector()
        
        # Smart supplier URL generation
        if supplier_url:
            self.url = supplier_url
        else:
            self.url = self._generate_supplier_url()
        
        self.suppliers = self.get_suppliers()
        self.produces = Commodities(ticker).produces
        self.requires = Commodities(ticker).requires
    
    def _generate_supplier_url(self) -> str:
        """
        Generate ImportYeti URL from company name.
        Converts "Apple Inc." → "https://www.importyeti.com/company/apple"
        """
        info = yf.Ticker(self.ticker).info
        company_name = info.get('longName', self.ticker)
        
        # Clean company name for URL
        clean_name = company_name.lower()
        
        # Remove common suffixes
        for suffix in [' inc.', ' inc', ' corporation', ' corp.', ' corp', 
                      ' llc', ' ltd.', ' ltd', ' company', ' co.', ' co']:
            clean_name = clean_name.replace(suffix, '')
        
        # Replace spaces with hyphens
        clean_name = clean_name.strip().replace(' ', '-')
        
        # Remove special characters
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-')
        
        return f"https://www.importyeti.com/company/{clean_name}"
    
    def get_location(self):
        info = yf.Ticker(self.ticker).info
        country = info.get("country", "Unknown")
        return [country]

    def get_sector(self):
        """Get sector and industry"""
        info = yf.Ticker(self.ticker).info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        
        print(f"📊 Sector: '{sector}', Industry: '{industry}'")
        
        return [sector, industry]
    
    def get_suppliers(self):
        try:
            return scrape_importyeti_suppliers(self.url)
        except Exception as e:
            print(f"⚠️  Could not scrape suppliers from {self.url}: {e}")
            return []
    
    def _compute_normalized_magnitude(self, magnitude, max_magnitude):
        """Compute normalized magnitude using log scaling"""
        if max_magnitude == 0:
            return 0.0
        return math.log(1 + magnitude) / math.log(1 + max_magnitude)
    
    def _add_normalized_properties(self, edges):
        """Add normalized_magnitude, direction, and weight to commodity edges."""
        
        # Group by commodity NAICS code
        produces_by_commodity = defaultdict(list)
        requires_by_commodity = defaultdict(list)
        
        for i, edge in enumerate(edges):
            if edge['type'] == 'produces':
                naics = edge['data']['naics_code']
                produces_by_commodity[naics].append((i, edge['magnitude']))
            elif edge['type'] == 'requires':
                naics = edge['data']['naics_code']
                requires_by_commodity[naics].append((i, edge['magnitude']))
        
        # Compute max magnitudes per commodity
        produces_max = {naics: max(mags for _, mags in edges_list) 
                       for naics, edges_list in produces_by_commodity.items()}
        requires_max = {naics: max(mags for _, mags in edges_list) 
                       for naics, edges_list in requires_by_commodity.items()}
        
        # Add normalized properties
        for i, edge in enumerate(edges):
            if edge['type'] == 'produces':
                naics = edge['data']['naics_code']
                max_mag = produces_max.get(naics, edge['magnitude'])
                
                normalized = self._compute_normalized_magnitude(edge['magnitude'], max_mag)
                direction = 1
                weight = normalized * direction
                
                edge['normalized_magnitude'] = normalized
                edge['direction_value'] = direction
                edge['weight'] = weight
                
            elif edge['type'] == 'requires':
                naics = edge['data']['naics_code']
                max_mag = requires_max.get(naics, edge['magnitude'])
                
                normalized = self._compute_normalized_magnitude(edge['magnitude'], max_mag)
                direction = -1
                weight = normalized * direction
                
                edge['normalized_magnitude'] = normalized
                edge['direction_value'] = direction
                edge['weight'] = weight
        
        return edges
    
    def edges(self):
        """
        Returns edges with DATA-DRIVEN metrics:
        - magnitude: Based on real data (market share, GDP, revenue)
        - relevance: Calculated confidence (not arbitrary)
        - correlation_strength: Market correlation
        - weight: Combined structural metric
        """
        edges_list = []
        
        # Country edges (DATA-DRIVEN)
        print("\n🌍 Computing country metrics...")
        for country in self.country:
            country_metrics = self.metrics.get_country_metrics(country)
            
            # Country → Company
            edge = {
                "type": "country",
                "magnitude": country_metrics['country_to_company_weight'],
                "relevance": country_metrics['country_to_company_confidence'],
                "direction": "country->company",
                "data": {
                    "country_name": country,
                    "relationship": "headquarters",
                    "impact_type": "regulatory_environment",

                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except Exception as e:
                    print(f"⚠️  Correlation failed: {e}")
                    edge['correlation_strength'] = country_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']
            
            edge['weight'] = country_metrics['country_to_company_weight']
            edge['confidence'] = country_metrics['country_to_company_confidence']
            edges_list.append(edge)
            
            # Company → Country
            edge = {
                "type": "country",
                "magnitude": country_metrics['company_to_country_weight'],
                "relevance": country_metrics['company_to_country_confidence'],
                "direction": "company->country",
                "data": {
                    "country_name": country,
                    "relationship": "economic_contributor",
                    "impact_type": "employment_tax_innovation",
                    "gdp_share_pct": country_metrics['gdp_share_pct'],
                    "company_revenue": country_metrics['company_revenue'],
                    "country_gdp": country_metrics['country_gdp']
                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except:
                    edge['correlation_strength'] = country_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']
            
            edge['weight'] = country_metrics['company_to_country_weight']
            edge['confidence'] = country_metrics['company_to_country_confidence']
            edges_list.append(edge)
        
        # Sector edges (DATA-DRIVEN)
        print("\n📊 Computing sector metrics...")
        sector_info = self.sector
        if len(sector_info) >= 1 and sector_info[0] != "Unknown":
            sector_name = sector_info[0]
            sector_metrics = self.metrics.get_sector_metrics(sector_name)
            
            # Sector → Company
            edge = {
                "type": "sector",
                "magnitude": sector_metrics['sector_to_company_weight'],
                "relevance": sector_metrics['sector_to_company_confidence'],
                "direction": "sector->company",
                "data": {
                    "sector_name": sector_name,
                    "classification_level": "sector",
                    "impact_type": "market_trends_regulations",
                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except:
                    edge['correlation_strength'] = sector_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']

            edge['weight'] = sector_metrics['sector_to_company_weight']
            edge['confidence'] = sector_metrics['sector_to_company_confidence']
            edges_list.append(edge)
            
            # Company → Sector
            edge = {
                "type": "sector",
                "magnitude": sector_metrics['company_to_sector_weight'],
                "relevance": sector_metrics['company_to_sector_confidence'],
                "direction": "company->sector",
                "data": {
                    "sector_name": sector_name,
                    "classification_level": "sector",
                    "impact_type": "innovation_disruption",
                    "sector_market_cap": sector_metrics['sector_market_cap']
                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except:
                    edge['correlation_strength'] = sector_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']
            
            edge['weight'] = sector_metrics['company_to_sector_weight']
            edge['confidence'] = sector_metrics['company_to_sector_confidence']
            edges_list.append(edge)
        
        # Industry edges (DATA-DRIVEN)
        print("\n🏭 Computing industry metrics...")
        if len(sector_info) >= 2 and sector_info[1] != "Unknown":
            industry_name = sector_info[1]
            industry_metrics = self.metrics.get_industry_metrics(industry_name)
            
            # Industry → Company
            edge = {
                "type": "industry",
                "magnitude": industry_metrics['industry_to_company_weight'],
                "relevance": industry_metrics['industry_to_company_confidence'],
                "direction": "industry->company",
                "data": {
                    "industry_name": industry_name,
                    "classification_level": "industry",
                    "impact_type": "competitive_dynamics",
                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except:
                    edge['correlation_strength'] = industry_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']
            
            edge['weight'] = industry_metrics['industry_to_company_weight']
            edge['confidence'] = industry_metrics['industry_to_company_confidence']
            edges_list.append(edge)
            
            # Company → Industry
            edge = {
                "type": "industry",
                "magnitude": industry_metrics['company_to_industry_weight'],
                "relevance": industry_metrics['company_to_industry_confidence'],
                "direction": "company->industry",
                "data": {
                    "industry_name": industry_name,
                    "classification_level": "industry",
                    "impact_type": "market_share_innovation",
                    "industry_market_cap": industry_metrics['industry_market_cap']
                }
            }
            
            if self.compute_correlations:
                try:
                    edge['correlation_strength'] = compute_correlation_strength(self.ticker, edge)
                except:
                    edge['correlation_strength'] = industry_metrics['correlation_strength']
            else:
                edge['correlation_strength'] = edge['magnitude'] * edge['relevance']
            
            edge['weight'] = industry_metrics['company_to_industry_weight']
            edge['confidence'] = industry_metrics['company_to_industry_confidence']
            edges_list.append(edge)
        
        # Supplier edges (already data-driven - shipment counts)
        print("\n📦 Processing supplier edges...")
        if self.suppliers:
            max_shipments = max([int(s.get('total_shipments', '0').replace(',', '')) 
                                for s in self.suppliers if s.get('total_shipments')] + [1])
            calculator = SupplierWeights(self.suppliers, self.ticker)
            enhanced_suppliers = calculator.calculate_all_weights()
            for supplier in enhanced_suppliers:
                try:
                    shipments = int(supplier.get('total_shipments', '0').replace(',', ''))
                except:
                    shipments = 0
                
                raw_magnitude = shipments
                normalized_magnitude = self._compute_normalized_magnitude(shipments, max_shipments)
                direction_value = 1
                weight = supplier["weight"]
                
                edges_list.append({
                    "type": "supplier",
                    "magnitude": raw_magnitude,
                    "normalized_magnitude": normalized_magnitude,
                    "direction_value": direction_value,
                    "weight": weight,
                    "relevance": supplier.get('confidence', 0.5),
                    "correlation_strength": weight * supplier.get('confidence', 0.5),
                    "direction": "supplier->company",
                    "data": {
                        "supplier_name": supplier.get('supplier_name', ''),
                        "supplier_url": supplier.get('supplier_url', ''),
                        "location": supplier.get('location', ''),
                        "total_shipments": supplier.get('total_shipments', ''),
                        "raw_shipment_count": shipments,
                        "product_description": supplier.get('product_description', ''),
                        "hs_codes": supplier.get('hs_codes', ''),
                        "impact_type": "supply_provision"
                    }
                })
        
        # Produces edges (data-driven from commodities engine)
        print("\n🏭 Processing commodity production edges...")
        for produce in self.produces:
            edges_list.append({
                "type": "produces",
                "magnitude": round(produce.get('production_value', 0), 6),
                "relevance": round(produce.get('confidence', 0.5), 4),
                "correlation_strength": round(produce.get('production_value', 0), 4),
                "direction": "company->commodity",
                "data": {
                    "naics_code": produce.get('naics_code', ''),
                    "description": produce.get('description', ''),
                    "commodity_type": "output"
                }
            })
        
        # Requires edges (data-driven from commodities engine)
        print("\n📥 Processing commodity requirement edges...")
        for require in self.requires:
            edges_list.append({
                "type": "requires",
                "magnitude": round(require.get('requirement_value', 0), 6),
                "relevance": round(require.get('confidence', 0.5), 4),
                "correlation_strength": round(require.get('requirement_value', 0), 4),
                "direction": "commodity->company",
                "data": {
                    "naics_code": require.get('naics_code', ''),
                    "description": require.get('description', ''),
                    "layer": require.get('layer', 'unknown'),
                    "commodity_type": "input"
                }
            })
        
        # Add normalized properties to commodity edges
        edges_list = self._add_normalized_properties(edges_list)
        
        print(f"\n✅ Generated {len(edges_list)} data-driven edges")
        return edges_list
    
    def to_json(self, filename=None, pretty=True):
        """Export edges to JSON file."""
        if filename is None:
            filename = f"{self.ticker}_edges.json"
        
        edges_data = {
            "ticker": self.ticker,
            "timestamp": yf.Ticker(self.ticker).info.get("longName", self.ticker),
            "total_edges": len(self.edges()),
            "edges": self.edges()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(edges_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(edges_data, f, ensure_ascii=False)
        
        print(f"✅ Exported {len(self.edges())} edges to {filename}")
        return filename
    
    def to_neo4j(self, 
                canonical_companies_csv="companies_filtered.csv",
                auto_canonicalize=False,
                min_occurrences=2,
                uri="bolt://127.0.0.1:7687", 
                user="neo4j", 
                password="myhome2911!"):
        
        to_neo4j_enhanced(
            self.edges(), 
            self.ticker,
            canonical_companies_csv=canonical_companies_csv,
            auto_canonicalize=auto_canonicalize,
            min_occurrences=min_occurrences,
            uri=uri,
            user=user,
            password=password
        )


# Example usage
if __name__ == "__main__":
    # Example 1: QuantumScape with explicit supplier URL
    edges = StaticEdges(
        "QS", 
        compute_correlations=True,
        supplier_url="https://www.importyeti.com/company/quantum-scape"
    )
    
    edges.to_neo4j(
        canonical_companies_csv="./data/companies_filtered.csv",
        auto_canonicalize=True,
        min_occurrences=2,
        password="myhome2911!",
    )
    
    # Example 2: Apple with auto-generated supplier URL
    # edges = StaticEdges("AAPL", compute_correlations=True)
    # edges.to_json("AAPL_data_driven_edges.json")