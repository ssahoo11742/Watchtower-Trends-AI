"""
Neo4j Connection and Health Test Script
Run this before using StaticEdges.to_neo4j() to verify your setup
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import sys

class Neo4jTester:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
    def test_connection(self):
        """Test if Neo4j is running and accessible"""
        print("\n" + "="*80)
        print("🔍 Testing Neo4j Connection")
        print("="*80)
        
        try:
            print(f"\n📡 Attempting to connect to: {self.uri}")
            print(f"👤 Username: {self.user}")
            
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Verify connectivity
            self.driver.verify_connectivity()
            
            print("✅ Connection successful!")
            return True
            
        except ServiceUnavailable as e:
            print(f"❌ Connection failed: Neo4j service is not available")
            print(f"   Error: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Make sure Neo4j is running (check Docker or Neo4j Desktop)")
            print("   2. Verify the URI is correct (default: bolt://localhost:7687)")
            print("   3. Check firewall settings")
            return False
            
        except AuthError as e:
            print(f"❌ Authentication failed: Invalid username or password")
            print(f"   Error: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check your username (default: neo4j)")
            print("   2. Verify your password")
            print("   3. Reset password in Neo4j Desktop or via cypher-shell")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def test_database_write(self):
        """Test if we can write to the database"""
        print("\n" + "="*80)
        print("✍️  Testing Database Write Permissions")
        print("="*80)
        
        if not self.driver:
            print("❌ No active connection. Run test_connection() first.")
            return False
        
        try:
            with self.driver.session() as session:
                # Create a test node
                result = session.run("""
                    CREATE (t:TestNode {name: 'Neo4j Test', timestamp: datetime()})
                    RETURN t.name as name, t.timestamp as timestamp
                """)
                
                record = result.single()
                print(f"✅ Successfully created test node: {record['name']}")
                print(f"   Timestamp: {record['timestamp']}")
                
                # Delete the test node
                session.run("MATCH (t:TestNode {name: 'Neo4j Test'}) DELETE t")
                print("✅ Successfully deleted test node")
                
                return True
                
        except Exception as e:
            print(f"❌ Write test failed: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check database permissions")
            print("   2. Ensure database is not in read-only mode")
            return False
    
    def test_database_read(self):
        """Test if we can read from the database"""
        print("\n" + "="*80)
        print("📖 Testing Database Read Permissions")
        print("="*80)
        
        if not self.driver:
            print("❌ No active connection. Run test_connection() first.")
            return False
        
        try:
            with self.driver.session() as session:
                # Get database info
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                
                for record in result:
                    print(f"✅ Successfully connected to Neo4j")
                    print(f"   Name: {record['name']}")
                    print(f"   Version: {record['versions'][0]}")
                    print(f"   Edition: {record['edition']}")
                
                return True
                
        except Exception as e:
            print(f"❌ Read test failed: {e}")
            return False
    
    def get_database_stats(self):
        """Get current database statistics"""
        print("\n" + "="*80)
        print("📊 Database Statistics")
        print("="*80)
        
        if not self.driver:
            print("❌ No active connection. Run test_connection() first.")
            return False
        
        try:
            with self.driver.session() as session:
                # Count nodes by label
                result = session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {})
                    YIELD value
                    RETURN label, value.count as count
                    ORDER BY count DESC
                """)
                
                print("\n📦 Nodes by Label:")
                total_nodes = 0
                for record in result:
                    count = record['count']
                    total_nodes += count
                    print(f"   {record['label']}: {count}")
                
                if total_nodes == 0:
                    print("   (Database is empty)")
                else:
                    print(f"\n   Total Nodes: {total_nodes}")
                
                # Count relationships by type
                result = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    CALL apoc.cypher.run('MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) as count', {})
                    YIELD value
                    RETURN relationshipType, value.count as count
                    ORDER BY count DESC
                """)
                
                print("\n🔗 Relationships by Type:")
                total_rels = 0
                for record in result:
                    count = record['count']
                    total_rels += count
                    print(f"   {record['relationshipType']}: {count}")
                
                if total_rels == 0:
                    print("   (No relationships)")
                else:
                    print(f"\n   Total Relationships: {total_rels}")
                
                return True
                
        except Exception as e:
            # APOC might not be installed, try simpler query
            try:
                with self.driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as node_count")
                    node_count = result.single()['node_count']
                    
                    result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                    rel_count = result.single()['rel_count']
                    
                    print(f"\n   Total Nodes: {node_count}")
                    print(f"   Total Relationships: {rel_count}")
                    
                    if node_count == 0:
                        print("\n   (Database is empty - ready for import!)")
                    
                    return True
            except Exception as e2:
                print(f"❌ Stats query failed: {e2}")
                return False
    
    def test_company_query(self, ticker="TSLA"):
        """Test if we can query for a specific company"""
        print("\n" + "="*80)
        print(f"🔎 Testing Query for Company: {ticker}")
        print("="*80)
        
        if not self.driver:
            print("❌ No active connection. Run test_connection() first.")
            return False
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Company {ticker: $ticker})
                    OPTIONAL MATCH (c)-[r]->(n)
                    RETURN c, count(r) as outgoing_relationships
                """, ticker=ticker)
                
                record = result.single()
                
                if record and record['c']:
                    company = record['c']
                    print(f"✅ Found company: {company.get('name', ticker)}")
                    print(f"   Ticker: {company.get('ticker')}")
                    print(f"   Market Cap: ${company.get('marketCap', 0):,.0f}")
                    print(f"   Sector: {company.get('sector')}")
                    print(f"   Industry: {company.get('industry')}")
                    print(f"   Outgoing Relationships: {record['outgoing_relationships']}")
                else:
                    print(f"⚠️  Company {ticker} not found in database")
                    print(f"   (This is expected if you haven't imported data yet)")
                
                return True
                
        except Exception as e:
            print(f"❌ Query test failed: {e}")
            return False
    
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            print("\n✅ Connection closed")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "🚀 " + "="*76)
        print("🚀  NEO4J HEALTH CHECK - Running All Tests")
        print("🚀 " + "="*76)
        
        results = {
            "connection": False,
            "read": False,
            "write": False,
            "stats": False,
            "query": False
        }
        
        # Test 1: Connection
        results["connection"] = self.test_connection()
        if not results["connection"]:
            print("\n❌ Connection failed. Cannot proceed with other tests.")
            self.close()
            return results
        
        # Test 2: Read
        results["read"] = self.test_database_read()
        
        # Test 3: Write
        results["write"] = self.test_database_write()
        
        # Test 4: Stats
        results["stats"] = self.get_database_stats()
        
        # Test 5: Query
        results["query"] = self.test_company_query()
        
        # Summary
        print("\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"   {test_name.capitalize():20s} {status}")
        
        print(f"\n   Score: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed! Your Neo4j setup is ready.")
            print("   You can now run StaticEdges.to_neo4j() safely.")
        else:
            print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        
        self.close()
        return results


def main():
    """Main function to run tests with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Neo4j connection and setup')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', default='password', help='Neo4j password')
    parser.add_argument('--ticker', default='TSLA', help='Ticker to test query')
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = Neo4jTester(uri=args.uri, user=args.user, password=args.password)
    
    # Run all tests
    tester.run_all_tests()


if __name__ == "__main__":
    # You can either run with command line arguments or edit these directly:
    
    # Option 1: Quick test with defaults
    tester = Neo4jTester(
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="myhome2911!"  # CHANGE THIS!
    )
    tester.run_all_tests()
    
    # Option 2: Run with command line arguments
    # Uncomment the line below and comment out the lines above
    # main()