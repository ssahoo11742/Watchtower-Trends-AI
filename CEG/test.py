"""
ALL-IN-ONE TEST SCRIPT
======================

This script will:
1. Test Neo4j connection
2. Upgrade your existing graph (if needed)
3. Run a signal propagation simulation
4. Show you the results

Just update the configuration section below and run:
    python test_everything.py
"""

from neo4j import GraphDatabase
import sys

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "myhome2911!"  # ⬅️ CHANGE THIS TO YOUR PASSWORD

TICKER = "QS"  # The company already in your database
EVENT_DESCRIPTION = "Major supplier disruption"
EVENT_SIGNAL = -0.7  # Negative signal (bad news)

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def test_connection():
    """Step 1: Test database connection"""
    print("\n" + "="*80)
    print("STEP 1: Testing Neo4j Connection")
    print("="*80)
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✅ Connected successfully!")
            print(f"   Database has {count} nodes")
            
            # Check if company exists
            result = session.run("MATCH (c:Company {ticker: $ticker}) RETURN c", ticker=TICKER)
            if result.single():
                print(f"✅ Found company: {TICKER}")
            else:
                print(f"❌ Company {TICKER} not found in database")
                print(f"   Available companies:")
                result = session.run("MATCH (c:Company) RETURN c.ticker as ticker LIMIT 10")
                for record in result:
                    print(f"      - {record['ticker']}")
                return False
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Neo4j running? Try: neo4j status")
        print("  2. Is the password correct?")
        print("  3. Is the URI correct? (default: bolt://127.0.0.1:7687)")
        return False


def check_schema():
    """Step 2: Check if schema needs upgrading"""
    print("\n" + "="*80)
    print("STEP 2: Checking Schema")
    print("="*80)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Check if propagation properties exist
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})
                RETURN 
                    c.risk_signal as has_signal,
                    c.base_weight as has_weight
            """, ticker=TICKER)
            
            record = result.single()
            if record and record['has_signal'] is not None:
                print("✅ Schema is already upgraded")
                print(f"   Company has risk_signal: {record['has_signal']}")
                print(f"   Company has base_weight: {record['has_weight']}")
                return True
            else:
                print("⚠️  Schema needs upgrading")
                return False
    
    finally:
        driver.close()


def upgrade_schema():
    """Step 3: Upgrade schema if needed"""
    print("\n" + "="*80)
    print("STEP 3: Upgrading Schema")
    print("="*80)
    
    try:
        from neo4j_enhanced_schema import upgrade_existing_graph
        
        print("Running schema upgrade...")
        upgrade_existing_graph(
            ticker=TICKER,
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        return True
        
    except ImportError:
        print("❌ Cannot import neo4j_enhanced_schema.py")
        print("   Make sure the file is in the same directory as this script")
        return False
    except Exception as e:
        print(f"❌ Upgrade failed: {e}")
        return False


def run_propagation():
    """Step 4: Run signal propagation"""
    print("\n" + "="*80)
    print("STEP 4: Running Signal Propagation")
    print("="*80)
    
    try:
        from signal_propagation import SignalPropagation
        
        propagator = SignalPropagation(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
        
        try:
            # Reset previous signals
            print("Resetting previous signals...")
            propagator.reset_signals()
            
            # Run simulation
            print(f"\n🎬 Simulating event: {EVENT_DESCRIPTION}")
            print(f"   Affected entity: {TICKER}")
            print(f"   Signal strength: {EVENT_SIGNAL}")
            
            affected_nodes = propagator.propagate_signal(
                source_ticker=TICKER,
                initial_signal=EVENT_SIGNAL,
                max_hops=3
            )
            
            # Get results
            print("\n" + "="*80)
            print("PROPAGATION RESULTS")
            print("="*80)
            
            most_affected = propagator.get_most_affected_nodes(top_n=15)
            
            if most_affected:
                print(f"\nTop {len(most_affected)} Most Affected Entities:\n")
                print(f"{'#':<4} {'Entity':<35} {'Type':<12} {'Signal':<10}")
                print("-" * 65)
                
                for i, node in enumerate(most_affected, 1):
                    signal_symbol = "📉" if node['signal'] < 0 else "📈"
                    print(f"{i:<4} {node['name']:<35} {node['type']:<12} {signal_symbol} {node['signal']:+.4f}")
                
                # Check for feedback loops
                print("\n" + "="*80)
                print("CHECKING FOR FEEDBACK LOOPS")
                print("="*80)
                
                loops = propagator.detect_reflexivity_loops(threshold=0.1)
                if loops:
                    print(f"\n⚠️  Detected {len(loops)} feedback loops:")
                    for i, loop in enumerate(loops[:5], 1):
                        print(f"  {i}. {loop['node_a']} ↔ {loop['node_b']}")
                        print(f"     Signals: {loop['signal_a']:.4f} / {loop['signal_b']:.4f}")
                else:
                    print("\n✅ No significant feedback loops detected")
                
                print("\n" + "="*80)
                print("SUCCESS!")
                print("="*80)
                print("\nNext steps:")
                print("  1. Open Neo4j Browser: http://localhost:7474")
                print("  2. Run this query to visualize:")
                print(f"\n     MATCH (n) WHERE n.risk_signal <> 0 RETURN n LIMIT 50\n")
                
                return True
            else:
                print("⚠️  No propagation occurred. This might mean:")
                print("   - The graph has no connections from this company")
                print("   - Edge weights are too low")
                print("   - Confidence thresholds filtered all edges")
                return False
                
        finally:
            propagator.close()
        
    except ImportError:
        print("❌ Cannot import signal_propagation.py")
        print("   Make sure the file is in the same directory as this script")
        return False
    except Exception as e:
        print(f"❌ Propagation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("NEO4J SIGNAL PROPAGATION - AUTOMATED TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Neo4j URI: {NEO4J_URI}")
    print(f"  User: {NEO4J_USER}")
    print(f"  Company: {TICKER}")
    print(f"  Event: {EVENT_DESCRIPTION}")
    
    # Step 1: Test connection
    if not test_connection():
        print("\n❌ Test failed at Step 1: Connection")
        sys.exit(1)
    
    # Step 2: Check schema
    needs_upgrade = not check_schema()
    
    # Step 3: Upgrade if needed
    if needs_upgrade:
        if not upgrade_schema():
            print("\n❌ Test failed at Step 3: Schema Upgrade")
            print("\nYou can try manually upgrading by running:")
            print("  python upgrade_graph.py")
            sys.exit(1)
    
    # Step 4: Run propagation
    if not run_propagation():
        print("\n❌ Test failed at Step 4: Propagation")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    main()