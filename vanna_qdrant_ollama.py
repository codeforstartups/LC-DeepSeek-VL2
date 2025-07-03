# my_vanna_qdrant_ollama.py

from vanna.qdrant import Qdrant_VectorStore
from vanna.ollama import Ollama
from qdrant_client import QdrantClient

class MyVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

# Initialize with your existing setup - Fixed configuration
vn = MyVanna(config={
    "client": QdrantClient(url="http://localhost:6333"),
    "fastembed_model": "BAAI/bge-small-en-v1.5",  # Specify embedding model
    "collection_name": "vyspr_collections",  # Use your existing collection
    "model": "deepseek-r1:1.5b",
    "base_url": "http://13.59.72.219:11434"  # Use base_url instead of ollama_host
})

print("✅ Vanna initialized with Qdrant + Ollama")
print(f"📊 Using existing collection: vyspr_collections")

# Test connection to Ollama with better error handling
try:
    print("🔄 Testing Ollama connection...")
    # Simple test that should work
    test_response = vn.generate_sql("SELECT 1 as test")
    print("✅ Ollama connection working!")
    print(f"Test SQL generated: {test_response}")
except Exception as e:
    print(f"❌ Ollama connection failed: {e}")
    print("🔧 Trying alternative connection...")
    try:
        # Try direct ollama test
        import ollama
        response = ollama.chat(
            model='deepseek-r1:1.5b',
            messages=[{'role': 'user', 'content': 'Generate SQL: SELECT 1'}],
            options={'host': 'http://13.59.72.219:11434'}
        )
        print("✅ Direct Ollama test successful!")
    except Exception as e2:
        print(f"❌ Direct test also failed: {e2}")

# Check your existing data in Qdrant
try:
    print("\n📊 Checking your existing data in Qdrant...")

    # Get collection info
    client = QdrantClient(url="http://localhost:6333")
    collection_info = client.get_collection("vyspr_collections")
    point_count = collection_info.points_count
    print(f"✅ Found {point_count} existing entries in vyspr_collections")

    if point_count > 0:
        # Sample some data to see what's there
        search_result = client.scroll(
            collection_name="vyspr_collections",
            limit=3,
            with_payload=True
        )[0]  # Get the points, not the next_page_offset

        print(f"📝 Sample data from your collection:")
        for i, point in enumerate(search_result):
            payload = point.payload if hasattr(point, 'payload') else "No payload"
            print(f"   {i+1}. ID: {point.id}, Payload: {payload}")

    # Check if vanna can access the data
    try:
        training_data = vn.get_training_data()
        print(f"🔍 Vanna can see {len(training_data)} training entries")
        if len(training_data) > 0:
            print("📚 Your training data is accessible!")
            for i, data in enumerate(training_data[:3]):  # Show first 3
                print(f"   {i+1}. {data}")
        else:
            print("⚠️  Vanna cannot access the data - might be a format issue")
    except Exception as e:
        print(f"❌ Error accessing training data through Vanna: {e}")

except Exception as e:
    print(f"❌ Error checking Qdrant data: {e}")

print(f"\n🚀 Ready to use! Try these commands:")
print(f"1. vn.ask('your question') - Generate and run SQL queries")
print(f"2. vn.generate_sql('your question') - Just generate SQL")
print(f"3. vn.get_training_data() - View your existing data")

print(f"\n💡 Example usage:")
print(f"sql = vn.generate_sql('Find all records containing financial information')")
print(f"result = vn.ask('What financial data do we have?')")

# If there's data but Vanna can't see it, let's try to investigate the format
if point_count > 0:
    print(f"\n🔧 Troubleshooting: Your collection has {point_count} points but Vanna might not recognize them.")
    print(f"   This could mean the data format needs adjustment for Vanna compatibility.")
    print(f"   Try: client.scroll('vyspr_collections', limit=1, with_payload=True) to inspect the format.")
