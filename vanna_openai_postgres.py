# vanna_openai_postgres.py

import os
from vanna.qdrant import Qdrant_VectorStore
from vanna.openai import OpenAI_Chat
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
# Set your OpenAI API key as an environment variable (recommended) or paste it directly.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY",)
if "YOUR_OPENAI_API_KEY" in OPENAI_API_KEY:
    print("❌ FATAL: OpenAI API key not found. Please set the variable.")
    exit()

# This class combines the OpenAI model with the Qdrant vector store
class VannaRAG(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def main():
    # Initialize Vanna with your configuration
    vn = VannaRAG(config={
        "client": QdrantClient(url="http://localhost:6333"),
        "collection_name": "vanna-postgres-openai",
        "api_key": OPENAI_API_KEY,
        "model": "gpt-4-turbo",
    })

    # Connect to PostgreSQL
    try:
        vn.connect_to_postgres(
            host="localhost",
            dbname="vyspr_dev",
            user="vyspr_user",
            password="vyspr_password",
            port="5432",
        )
        print("✅ Connected to PostgreSQL.")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return

    # Automatically train on the entire database schema if no training data exists
    if not vn.get_training_data().empty:
        print("✅ Training data found. Skipping training.")
    else:
        print("⏳ No training data found. Training on the entire database schema...")

        # Get all table DDLs from the public schema
        df_information_schema = vn.run_sql(
            "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
        )

        # Group by table and generate CREATE TABLE statements
        for table_name, group in df_information_schema.groupby('table_name'):
            columns_ddl = [f"    \"{row['column_name']}\" {row['data_type'].upper()}" for _, row in group.iterrows()]
            ddl_string = f"CREATE TABLE public.\"{table_name}\" (\n" + ",\n".join(columns_ddl) + "\n);"
            vn.train(ddl=ddl_string)

        print("✅ Training complete.")

    # Ask a question
    print("\n🚀 Vanna is ready. Asking a question...")
    try:
        question = "total prchased number of items and also i need that numbers"
        print(f"\n🤔 Question: {question}")

        # Using the more robust two-step process to avoid the '.ask()' bug
        sql = vn.generate_sql(question=question)

        if sql:
            print(f"\n🤖 Generated SQL:\n{sql}")
            df = vn.run_sql(sql=sql)
            print("\n✅ Answer:")
            print(df)
        else:
            print("\n⚠️ The model was unable to generate SQL for the question.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == '__main__':
    main()
