# vanna_api.py
import os
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from vanna.qdrant import Qdrant_VectorStore
from vanna.ollama import Ollama # Changed from OpenAI
from qdrant_client import QdrantClient

# --- Global Vanna Instance ---
# This will be initialized on startup
vn = None

# --- API Configuration ---
app = FastAPI(
    title="Vanna Text-to-SQL API (Ollama Backend)",
    description="An API to convert natural language questions to SQL queries using a self-hosted Ollama model.",
    version="1.1.0"
)

# --- Pydantic Models for Request and Response ---
class SQLRequest(BaseModel):
    question: str = Field(..., example="How many users are there?")

class SQLResponse(BaseModel):
    question: str
    sql_query: str
    result: List[Dict[str, Any]]
    generation_time_seconds: float

# --- Vanna Initialization on Application Startup ---
@app.on_event("startup")
async def startup_event():
    global vn

    # This class now combines Ollama with the Qdrant vector store
    class VannaRAG(Qdrant_VectorStore, Ollama):
        def __init__(self, config=None):
            Qdrant_VectorStore.__init__(self, config=config)
            Ollama.__init__(self, config=config)

    # Updated configuration for Ollama
    vn = VannaRAG(config={
        "client": QdrantClient(url="http://localhost:6333"),
        "collection_name": "vanna-postgres-langchain-dev", # Keeping the same collection
        "model": "deepseek-r1:1.5b",
        "base_url": "http://13.59.72.219:11434",
    })

    # Connect to the database
    try:
        vn.connect_to_postgres(
            host="localhost",
            dbname="langchain_dev",
            user="langchain_user",
            password="langchain_password",
            port="5432",
        )
        print("✅ Successfully connected to PostgreSQL and initialized Vanna with Ollama.")
    except Exception as e:
        print(f"❌ FATAL: Could not connect to PostgreSQL. {e}")
        # In a real app, you might want a more robust retry logic
        raise RuntimeError(f"Database connection failed: {e}")

    # --- Automated Training ---
    # Ensures Vanna is trained on the schema and has our custom examples.
    training_data = vn.get_training_data()

    # If the training data is empty, it's the first run for this collection.
    # We need to train it on the DDL and our custom SQL examples.
    if training_data.empty:
        print("⏳ No training data found. Performing initial training...")

        # 1. Train on database schema (DDL)
        print("   - Training on database schema (DDL)...")
        df_information_schema = vn.run_sql(
            "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
        )
        for table_name, group in df_information_schema.groupby('table_name'):
            columns_ddl = [f"    \"{row['column_name']}\" {row['data_type'].upper()}" for _, row in group.iterrows()]
            ddl_string = f"CREATE TABLE public.\"{table_name}\" (\n" + ",\n".join(columns_ddl) + "\n);"
            vn.train(ddl=ddl_string)
        print("   ✅ DDL training complete.")

        # 2. Add specific Question-SQL pairs for guidance
        print("   - Adding custom question-SQL training pairs...")
        vn.train(
            question="How many users are there?",
            sql="SELECT COUNT(*) FROM users;"
        )
        vn.train(
            question="What are the names of all the tables?",
            sql="SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )
        # Add more examples here for your most common questions
        print("   ✅ Custom training complete.")
        print("✅ Initial training complete.")
    else:
        print("✅ Training data found. Skipping initial training.")


# --- API Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "vanna_initialized": vn is not None}

@app.post("/sql_query", response_model=SQLResponse)
async def generate_sql_query(request: SQLRequest):
    """
    Accepts a natural language question and returns the generated SQL and its result.
    """
    if vn is None:
        raise HTTPException(status_code=503, detail="Vanna service is not available or still starting up.")

    try:
        start_time = time.time()

        # Use the robust two-step process
        sql = vn.generate_sql(question=request.question)

        if not sql:
            raise HTTPException(status_code=404, detail="The model was unable to generate a SQL query for the given question.")

        df_result = vn.run_sql(sql=sql)

        generation_time = time.time() - start_time

        # Convert DataFrame to a list of dictionaries for JSON compatibility
        result_dict = df_result.to_dict(orient='records') if df_result is not None else []

        return SQLResponse(
            question=request.question,
            sql_query=sql,
            result=result_dict,
            generation_time_seconds=round(generation_time, 4)
        )

    except Exception as e:
        print(f"❌ An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
