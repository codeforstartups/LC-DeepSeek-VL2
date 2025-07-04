# vanna_postgre_ollam.py

from vanna.qdrant import Qdrant_VectorStore
from vanna.ollama import Ollama
from qdrant_client import QdrantClient

class MyVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        # A new collection will be created for this specific database
        collection_name = config.get('collection_name', 'vanna-postgres-vyspr-dev')

        # Override the collection name in the config
        qdrant_config = config.copy()
        qdrant_config['collection_name'] = collection_name

        Qdrant_VectorStore.__init__(self, config=qdrant_config)
        Ollama.__init__(self, config=config)

# --- Configuration ---
# Your PostgreSQL connection details
POSTGRES_CONFIG = {
    "host": "localhost",  # Or your DB host
    "dbname": "langchain_dev",
    "user": "langchain_user",
    "password": "langchain_password",
    "port": "5432"
}

# Your Ollama and Qdrant configuration
VANNA_CONFIG = {
    "client": QdrantClient(url="http://localhost:6333"),
    "collection_name": "vanna-postgres-langchain-dev",
    "model": "deepseek-r1:1.5b",
    "base_url": "http://13.59.72.219:11434"
}
# --- End Configuration ---

# Initialize Vanna
vn = MyVanna(config=VANNA_CONFIG)

# Connect to PostgreSQL
print(f"🔄 Connecting to PostgreSQL database '{POSTGRES_CONFIG['dbname']}'...")
try:
    vn.connect_to_postgres(
        host=POSTGRES_CONFIG['host'],
        dbname=POSTGRES_CONFIG['dbname'],
        user=POSTGRES_CONFIG['user'],
        password=POSTGRES_CONFIG['password'],
        port=POSTGRES_CONFIG['port'],
    )
    print("✅ Successfully connected to PostgreSQL!")
except Exception as e:
    print(f"❌ Failed to connect to PostgreSQL: {e}")
    print("Please ensure the database is running and credentials are correct.")
    exit()

# --- Training ---
# This part trains Vanna on your database's schema.
# It runs once and stores the schema information in Qdrant.
print("\n🔄 Checking for existing training data...")
training_data = vn.get_training_data()

if training_data is not None and len(training_data) > 0:
    print(f"✅ Found {len(training_data)} existing training entries. Skipping training.")
else:
    print("❌ No training data found. Training Vanna on your schema...")
    try:
        # This query will extract the schema from all of your public tables
        print("📜 Step 1: Running SQL to get schema for all public tables...")
        sql_query = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
        print(f"   Query: {sql_query}")
        df_information_schema = vn.run_sql(sql_query)

        if df_information_schema is None or df_information_schema.empty:
             raise ValueError("Failed to retrieve schema information from the database.")

        print(f"✅ Successfully retrieved schema for {len(df_information_schema.groupby('table_name'))} tables.")

        # -- START AUTOMATED DDL TRAINING --
        print("\n📜 Step 2: Automatically generating and training on DDL for each table...")

        # Group by table name and generate a DDL for each
        for table_name, group in df_information_schema.groupby('table_name'):
            print(f"   - Training on table: {table_name}")

            columns_ddl = []
            for index, row in group.iterrows():
                # Sanitize data types for DDL
                data_type = row['data_type'].upper()
                if 'timestamp' in data_type.lower():
                    data_type = 'TIMESTAMP'

                columns_ddl.append(f"    \"{row['column_name']}\" {data_type}")

            ddl_string = f"CREATE TABLE public.\"{table_name}\" (\n" + ",\n".join(columns_ddl) + "\n);"

            # Train on the generated DDL
            vn.train(ddl=ddl_string)

        print("✅ Training complete for all tables!")
        # -- END AUTOMATED DDL TRAINING --

    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check database permissions and table accessibility.")
        exit()
# --- End Training ---


# --- Ready to Ask Questions ---
print("\n🚀 Vanna is ready to answer your questions!")
print("Example: vn.ask('What tables do I have?')")
print("Example: vn.ask('Show me the first 5 rows of the users table')")

# Example question to test the setup
print("\n🧪 Running a test question...")
try:
    test_question = "What are the tables in the database?"
    print(f">>> Asking: '{test_question}'")
    answer_df = vn.ask(test_question)

    if answer_df is not None and not answer_df.empty:
        print("\n✅ Test successful! Answer:")
        print(answer_df)
    else:
        print("⚠️ Test question ran, but returned no data.")

except Exception as e:
    print(f"❌ Error during test question: {e}")