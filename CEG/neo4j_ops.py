from neo4j import GraphDatabase
import yfinance as yf

def to_neo4j(edges, ticker, uri="bolt://localhost:7687", user="neo4j", password="password"):
    """
    Export edges to Neo4j graph database with complete metadata.

    Args:
        edges: List of edges to export (already a list!)
        ticker: Company ticker symbol
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
    
    Edge Properties Stored:
        - magnitude: Impact strength (0.0-1.0)
        - relevance: Confidence score (0.0-1.0)
        - correlation: "positive" or "negative" (how entities move together)
        - type: Edge category for filtering
        - Additional edge-specific data (shipments, HS codes, etc.)
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Get company info for central node
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
            
            # Create central company node
            session.run("""
                MERGE (c:Company {ticker: $ticker})
                SET c.name = $name,
                    c.marketCap = $marketCap,
                    c.sector = $sector,
                    c.industry = $industry
            """, 
            ticker=ticker,
            name=company_name,
            marketCap=info.get("marketCap", 0),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"))
            
            # Process each edge
            for edge in edges:
                edge_type = edge['type']
                direction = edge['direction']
                magnitude = edge['magnitude']
                relevance = edge['relevance']
                correlation = edge.get('correlation', 'positive')
                data = edge['data']
                
                # COUNTRY EDGES
                if edge_type == 'country':
                    # Create/merge country node
                    session.run("""
                        MERGE (n:Country {name: $country_name})
                        SET n.relationship = $relationship
                    """, 
                    country_name=data['country_name'],
                    relationship=data['relationship'])
                    
                    if direction == "country->company":
                        # Country influences company (regulations, economy, etc.)
                        session.run("""
                            MATCH (country:Country {name: $country_name})
                            MATCH (company:Company {ticker: $ticker})
                            MERGE (country)-[r:INFLUENCES]->(company)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.type = 'country'
                        """,
                        country_name=data['country_name'],
                        ticker=ticker,
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''))
                    else:
                        # Company impacts country (jobs, taxes, innovation)
                        session.run("""
                            MATCH (company:Company {ticker: $ticker})
                            MATCH (country:Country {name: $country_name})
                            MERGE (company)-[r:IMPACTS]->(country)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.market_cap = $market_cap,
                                r.type = 'country'
                        """,
                        ticker=ticker,
                        country_name=data['country_name'],
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''),
                        market_cap=data.get('market_cap', 0))
                
                # SECTOR EDGES
                elif edge_type == 'sector':
                    # Create/merge sector node
                    session.run("""
                        MERGE (n:Sector {name: $sector_name})
                        SET n.classification_level = $classification_level
                    """,
                    sector_name=data['sector_name'],
                    classification_level=data['classification_level'])
                    
                    if direction == "sector->company":
                        # Sector influences company (market trends, regulations)
                        session.run("""
                            MATCH (sector:Sector {name: $sector_name})
                            MATCH (company:Company {ticker: $ticker})
                            MERGE (sector)-[r:INFLUENCES]->(company)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.type = 'sector'
                        """,
                        sector_name=data['sector_name'],
                        ticker=ticker,
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''))
                    else:
                        # Company impacts sector (innovation, disruption)
                        session.run("""
                            MATCH (company:Company {ticker: $ticker})
                            MATCH (sector:Sector {name: $sector_name})
                            MERGE (company)-[r:IMPACTS]->(sector)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.market_cap = $market_cap,
                                r.type = 'sector'
                        """,
                        ticker=ticker,
                        sector_name=data['sector_name'],
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''),
                        market_cap=data.get('market_cap', 0))
                
                # INDUSTRY EDGES (Competitive dynamics - NEGATIVE correlation!)
                elif edge_type == 'industry':
                    # Create/merge industry node
                    session.run("""
                        MERGE (n:Industry {name: $industry_name})
                        SET n.classification_level = $classification_level
                    """,
                    industry_name=data['industry_name'],
                    classification_level=data['classification_level'])
                    
                    if direction == "industry->company":
                        # Industry influences company (competition, market share battles)
                        session.run("""
                            MATCH (industry:Industry {name: $industry_name})
                            MATCH (company:Company {ticker: $ticker})
                            MERGE (industry)-[r:INFLUENCES]->(company)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.type = 'industry'
                        """,
                        industry_name=data['industry_name'],
                        ticker=ticker,
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''))
                    else:
                        # Company impacts industry (gains/losses in market share)
                        session.run("""
                            MATCH (company:Company {ticker: $ticker})
                            MATCH (industry:Industry {name: $industry_name})
                            MERGE (company)-[r:IMPACTS]->(industry)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.impact_type = $impact_type,
                                r.market_cap = $market_cap,
                                r.type = 'industry'
                        """,
                        ticker=ticker,
                        industry_name=data['industry_name'],
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        impact_type=data.get('impact_type', ''),
                        market_cap=data.get('market_cap', 0))
                
                # SUPPLIER EDGES (Supply chain dependencies)
                elif edge_type == 'supplier':
                    # Create/merge supplier node with detailed info
                    session.run("""
                        MERGE (n:Supplier {name: $supplier_name})
                        SET n.url = $supplier_url,
                            n.location = $location,
                            n.product_description = $product_description,
                            n.hs_codes = $hs_codes
                    """,
                    supplier_name=data['supplier_name'],
                    supplier_url=data.get('supplier_url', ''),
                    location=data.get('location', ''),
                    product_description=data.get('product_description', ''),
                    hs_codes=data.get('hs_codes', ''))
                    
                    if direction == "supplier->company":
                        # Supplier provides inputs to company
                        session.run("""
                            MATCH (supplier:Supplier {name: $supplier_name})
                            MATCH (company:Company {ticker: $ticker})
                            MERGE (supplier)-[r:SUPPLIES]->(company)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.total_shipments = $total_shipments,
                                r.impact_type = $impact_type,
                                r.type = 'supplier'
                        """,
                        supplier_name=data['supplier_name'],
                        ticker=ticker,
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        total_shipments=data.get('total_shipments', ''),
                        impact_type=data.get('impact_type', ''))
                    else:
                        # Company creates demand for supplier
                        session.run("""
                            MATCH (company:Company {ticker: $ticker})
                            MATCH (supplier:Supplier {name: $supplier_name})
                            MERGE (company)-[r:DEMANDS_FROM]->(supplier)
                            SET r.magnitude = $magnitude,
                                r.relevance = $relevance,
                                r.correlation = $correlation,
                                r.total_shipments = $total_shipments,
                                r.impact_type = $impact_type,
                                r.type = 'supplier'
                        """,
                        ticker=ticker,
                        supplier_name=data['supplier_name'],
                        magnitude=magnitude,
                        relevance=relevance,
                        correlation=correlation,
                        total_shipments=data.get('total_shipments', ''),
                        impact_type=data.get('impact_type', ''))
                
                # PRODUCES EDGES (Output commodities)
                elif edge_type == 'produces':
                    # Create/merge commodity node
                    session.run("""
                        MERGE (n:Commodity {naics_code: $naics_code})
                        SET n.description = $description,
                            n.commodity_type = $commodity_type
                    """,
                    naics_code=data['naics_code'],
                    description=data.get('description', ''),
                    commodity_type=data.get('commodity_type', ''))
                    
                    # Company produces commodity
                    session.run("""
                        MATCH (company:Company {ticker: $ticker})
                        MATCH (commodity:Commodity {naics_code: $naics_code})
                        MERGE (company)-[r:PRODUCES]->(commodity)
                        SET r.magnitude = $magnitude,
                            r.relevance = $relevance,
                            r.correlation = $correlation,
                            r.type = 'produces'
                    """,
                    ticker=ticker,
                    naics_code=data['naics_code'],
                    magnitude=magnitude,
                    relevance=relevance,
                    correlation=correlation)
                
                # REQUIRES EDGES (Input commodities)
                elif edge_type == 'requires':
                    # Create/merge commodity node
                    session.run("""
                        MERGE (n:Commodity {naics_code: $naics_code})
                        SET n.description = $description,
                            n.commodity_type = $commodity_type
                    """,
                    naics_code=data['naics_code'],
                    description=data.get('description', ''),
                    commodity_type=data.get('commodity_type', ''))
                    
                    # Commodity required by company
                    session.run("""
                        MATCH (commodity:Commodity {naics_code: $naics_code})
                        MATCH (company:Company {ticker: $ticker})
                        MERGE (commodity)-[r:REQUIRED_BY]->(company)
                        SET r.magnitude = $magnitude,
                            r.relevance = $relevance,
                            r.correlation = $correlation,
                            r.layer = $layer,
                            r.type = 'requires'
                    """,
                    naics_code=data['naics_code'],
                    ticker=ticker,
                    magnitude=magnitude,
                    relevance=relevance,
                    correlation=correlation,
                    layer=data.get('layer', 'unknown'))
            
            print(f"✅ Successfully exported {len(edges)} edges to Neo4j")
            print(f"   - Nodes: Company, Country, Sector, Industry, Supplier, Commodity")
            print(f"   - Relationships: INFLUENCES, IMPACTS, SUPPLIES, DEMANDS_FROM, PRODUCES, REQUIRED_BY")
            print(f"   - All edges include: magnitude, relevance, correlation")
            return True
            
    except Exception as e:
        print(f"❌ Error exporting to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        driver.close()